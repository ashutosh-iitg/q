from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Annotated
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from langchain_core.tracers.langchain import wait_for_all_tracers
from openai import APIError
from pydantic import ValidationError
from psycopg_pool import PoolTimeout

from api.schemas import MessageIn, MessageOut, PathId
from app.agent.graph import build_graph
from app.agent.handoff import HandoffError, read_note
from app.agent.loop import ModelOutputError
from app.config import chat_model, handoff_dir, require_env
from app.demo import demo_accounts
from app.consolidation import Consolidate, consolidate_idle_sessions, consolidate_session, llm_consolidator
from app.store import SessionOwnershipError, Store, TurnConflictError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = Store.connect(require_env("DATABASE_URL"))
    try:
        store.init_schema()
        for account in demo_accounts():
            store.upsert_members(account["household_id"], account["members"])
        app.state.store = store
        mini = chat_model()
        app.state.consolidate = llm_consolidator(mini)
        app.state.handoff_dir = handoff_dir()
        app.state.agent = build_graph(store, mini, handoff_dir=app.state.handoff_dir)
        yield
    finally:
        store.close()
        wait_for_all_tracers()


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_consolidator(request: Request) -> Consolidate:
    return request.app.state.consolidate


StoreDep = Annotated[Store, Depends(get_store)]
ConsolidatorDep = Annotated[Consolidate, Depends(get_consolidator)]


def create_app() -> FastAPI:
    app = FastAPI(title="Q", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/demo/accounts")
    def accounts() -> list[dict]:
        return demo_accounts()

    # Sync handlers: graph.invoke and the store block, so FastAPI runs them in its threadpool.
    # Background tasks run after the response is sent, so consolidation never delays a reply.
    @app.post("/households/{household_id}/sessions/{session_id}/messages")
    def post_message(
        household_id: PathId,
        session_id: PathId,
        message: MessageIn,
        background_tasks: BackgroundTasks,
        request: Request,
        store: StoreDep,
        consolidate: ConsolidatorDep,
    ) -> MessageOut:
        turn_id = message.turn_id or uuid4().hex
        if message.profile_id and not store.has_member(household_id, message.profile_id):
            raise HTTPException(status_code=404, detail="Profile not found for this household")
        try:
            result = request.app.state.agent.invoke(
                {"household_id": household_id, "session_id": session_id, "turn_id": turn_id,
                 "text": message.text, "profile_id": message.profile_id},
                config={
                    "run_name": "q.turn",
                    "tags": ["turn"],
                    "metadata": {"household_id": household_id, "session_id": session_id, "turn_id": turn_id},
                },
            )
        except SessionOwnershipError as exc:
            raise HTTPException(status_code=404, detail="Session not found for this household") from exc
        except TurnConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (APIError, ModelOutputError, ValidationError) as exc:
            logger.warning("Q model failure: %s (status=%s, code=%s)",
                           type(exc).__name__, getattr(exc, "status_code", None), getattr(exc, "code", None))
            raise HTTPException(status_code=502, detail=(
                "Q could not generate a reply. Retry with the same turn_id. If symptoms are severe or "
                "getting worse, call your local emergency number now."
            )) from exc
        except HandoffError as exc:
            logger.error("Doctor handoff failed: %s", exc)
            raise HTTPException(status_code=503, detail=(
                "Q could not hand this case to a doctor. Retry with the same turn_id. If symptoms are "
                "severe or getting worse, call your local emergency number now."
            )) from exc
        except PoolTimeout as exc:
            raise HTTPException(status_code=503, detail="Q's database is temporarily unavailable") from exc
        if result["is_new_session"]:
            # Lazy trigger: a new session closes and consolidates the household's abandoned ones.
            background_tasks.add_task(consolidate_idle_sessions, store, consolidate, household_id, session_id)
        return MessageOut(turn_id=turn_id, decision=result["decision"], reply=result["reply"])

    @app.get("/households/{household_id}/handoffs/{turn_id}")
    def get_handoff(household_id: PathId, turn_id: PathId, request: Request) -> dict:
        """The doctor intake note a turn saved; the demo shows it in place of a real handoff."""
        note = read_note(request.app.state.handoff_dir, household_id, turn_id)
        if note is None:
            raise HTTPException(status_code=404, detail="No intake note for this turn")
        return note

    @app.post("/households/{household_id}/sessions/{session_id}/close", status_code=202)
    def close_session(
        household_id: PathId,
        session_id: PathId,
        background_tasks: BackgroundTasks,
        store: StoreDep,
        consolidate: ConsolidatorDep,
    ) -> dict[str, str]:
        if not store.close_session(household_id, session_id):
            raise HTTPException(status_code=404, detail="No active session with this ID for this household")
        background_tasks.add_task(consolidate_session, store, consolidate, session_id)
        return {"status": "consolidating"}

    return app


app = create_app()
