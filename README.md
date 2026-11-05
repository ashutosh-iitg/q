# Q

A lightweight family health agent for the assignment. One LangGraph run handles each
message: load PostgreSQL context → screen for red flags alongside member resolution →
resolve the episode using that member → join and validate → observe the message against that episode →
apply code policy → return fixed escalation text or generate a reply → save the turn atomically.

## Run locally

For a one-command demo, start Docker, configure `.env` as below, then run `./demo.sh`.
The script installs locked dependencies, starts local PostgreSQL, waits for the API to seed
the 10 demo accounts, and starts Streamlit at http://127.0.0.1:8501. It requires free ports
5432 (PostgreSQL), 8000 (API), and 8501 (Streamlit); an existing Compose database is reused.
Ctrl+C stops the API and UI. Database data is retained; use `docker compose stop postgres`
to stop PostgreSQL. If `.env` is missing, the script creates it and asks you to fill in the
model settings before rerunning.

Requires Python 3.13+, uv, Docker, and an OpenAI API key with access to a model that
supports structured output.

```sh
cp .env.example .env
# Fill POSTGRES_PASSWORD, matching DATABASE_URL, OPENAI_API_KEY and Q_MODEL_MINI.
uv sync
# Compose reads .env for PostgreSQL's password.
docker compose up -d
uv run --env-file .env uvicorn api.main:app --reload
```

The app creates the schema, upgrades the earlier global turn-key constraint, and
upserts 10 synthetic demo families (35 members) at startup. The original `demo`
household remains the Sharma family: Ravi, Asha, Maya and Ashwin.
Open http://localhost:8000/docs to explore the API.

## Chat interface

With the API running, open a second terminal:

```sh
uv run --env-file .env streamlit run streamlit_app.py
```

Open http://localhost:8501 and choose an **Account** (family), then a **Profile**
(family member). Ask questions using the chat input. The selected profile gives Q
the default subject for personal reports; naming another member lets you discuss
them instead. Each account/profile selection keeps a separate session and transcript
while the page is open. Switching back resumes that chat. **New chat** closes the
previous session for background consolidation and starts a fresh one. Family memory
is shared across that household's sessions. The sidebar exports the current chat as
Markdown or JSON (the JSON keeps any doctor intake notes). A failed message can be retried using
the same turn ID, including after a response timeout.

The UI calls FastAPI over HTTP. Set `Q_API_URL` if the API is hosted somewhere other
than `http://localhost:8000`. The UI does not need database access or an OpenAI key.
Browser refresh starts fresh UI sessions; saved conversations remain in PostgreSQL.
The selectors are for this local demo and do not implement account authentication.

`fixtures/demo_families.json` defines all 10 families, each with 2–5 members.
They cover infants, children, teenagers, adults and older adults, including three
families with no known conditions and others with conditions such as asthma,
diabetes, hypertension and kidney disease. All people and histories are synthetic.
`GET /demo/accounts` exposes this seeded catalog to the UI. Message requests accept
an optional `profile_id`, which must belong to the requested household.

UI behavior is checked with Streamlit's
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
including account/profile switching, retries and closing a chat.

```sh
curl http://localhost:8000/households/demo/sessions/first/messages \
  -H 'Content-Type: application/json' \
  -d '{"text":"Why is sleep important?","turn_id":"question-1"}'
```

Supply a new `turn_id` per message. Retrying the same session/key/text returns the
persisted response without model calls. Reusing that key with different text returns
409; another session may use the same key. Omitting the key generates one, returned
in the response. A model failure returns 502; retry with the same key.

## Code layout

```text
app/
  graph/        the turn graph: build.py (wiring), nodes.py, replies.py, routing.py
  agents/       the model-driven steps and what they share
    loop.py       bounded lookup-then-answer loop every agent runs
    tools.py      read-only, household-scoped lookups
    resolvers.py  member and episode resolution
    intake/       observe.py (map the message to the protocol), merge.py (findings), questions.py (next question)
    handoff/      note.py (doctor intake note), draft.py (its schema), faithfulness.py (evidence checks)
    context.py, extraction.py, evidence.py
  prompts/      one system prompt module per role, plus shared rules and tone
  models/       outputs.py (model output schemas), records.py (stored records), turn.py (graph state)
  memory/       store.py (Postgres), sql.py, rows.py, writes.py, schema.sql, consolidation.py
  triage/       engine.py (sufficiency), protocol.py, policy.py (disposition rules), protocols/*.yaml
  config.py, extensions.py, demo.py
api/            FastAPI app and request/response schemas
ui/             Streamlit client helpers and transcript export
streamlit_app.py  the chat UI
scripts/        smoke test, terminal debug chat, scenario runner
tests/          pytest suite (fake model; Postgres tests skip without DATABASE_URL)
```

## Current implementation

- Prompts live in `app/prompts/` (one module per role); model output schemas in `app/models/outputs.py`.
  The member and episode agents (`app/agents/resolvers.py`) call read-only, household-scoped
  tools (`app/agents/tools.py`). When either can't decide, Q asks a clarifying question
  (`ask_member` or `ask_episode`) using the agent's instructions.
  `Q_MODEL_MINI` serves every model step and consolidation; optional `Q_REASONING_EFFORT`
  (e.g. `low`, `high`) sets its reasoning effort.
- Symptom intake follows triage protocols (`app/triage/protocols/*.yaml`, synthetic placeholders
  in the format of licensed guidelines). A disposition is declared once it is *sufficient*: a
  positive answer at its level and every more urgent question ruled out. Until then code picks the
  next finding to settle (most urgent first) and the intake agent words one question for it, using
  the protocol as guidance and the conversation as context ("you" for a self-report, a clarification
  answered before re-asking). Wording that isn't a single short question falls back to the protocol text.
  The generic fallback routes to doctor review after negative checks; it cannot authorize home care.
  The doctor note lists only what was discussed: each history sentence must quote the user's own words
  (Q's questions and bare "no"/"yes" replies are not evidence; unsupported sentences are dropped and
  counted), denials come from code with the question Q asked, and every item says which conversation
  reported it and when.
  Doctor decisions run a
  handoff agent that calls `escalate_to_doctor`, saving a JSON intake note under `handoffs/`
  (`Q_HANDOFF_DIR`).
- Code selects emergency, doctor, clarification, general answer or home monitoring.
  Emergency and doctor responses are fixed. Synthetic policy defaults live in
  `app/triage/policy.py`; they are not clinically validated.
- PostgreSQL stores messages, member facts, sessions and episodes. Follow-ups merge
  slots into the selected episode. Reported resolution and recurrence stay distinct.
- Consolidation runs after session closure or a new session's idle-session check.
  Recovered facts require a verbatim user quote and the message's recorded member.
  Provenance checks do not prove semantic correctness or resolve every mixed-member ambiguity.
- `/health` is a process health endpoint. This local assignment harness has no
  authentication or real doctor integration; the fixed transfer text is the
  assignment's simulated handoff.

## Verify

```sh
uv run --env-file .env pytest -q
# Calls the real model with a synthetic general question; checks persistence and replay.
uv run --env-file .env python -m scripts.smoke
```

To step through a turn with `pdb`, run the graph in-process from a terminal chat
(it loads `.env` itself); put `breakpoint()` in any node:

```sh
python -m scripts.debug_chat [household_id] [profile_id]   # defaults: demo, demo-ravi
```

Tests exercise PostgreSQL and the real graph/API with a fake model, including
follow-up state, fixed escalation, idempotency, provenance and retry after failure.
Without `DATABASE_URL`, database tests skip. The smoke command makes paid API calls
and leaves a small synthetic household/conversation in the local database.

## Observability

Q uses LangSmith. Add these settings to `.env` (choose the endpoint matching your
LangSmith account's region):

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=q-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Restart the API, or run the smoke command above, then open the `q-dev` tracing
project in LangSmith. Every API message creates a `q.turn` trace containing the
graph nodes, model prompts/results, token usage, latency and errors. Metadata
includes `household_id`, `session_id` (LangSmith thread grouping) and `turn_id`.
Each model call is named for its agent and step (`MemberResolution · call 1`,
`Observation · call 1`, `Screen`, `reply`), and the turn's output is a summary: decision,
reason, urgency, member, episode, protocol, reply and replay status. Tests never upload
traces, and `scripts/debug_chat.py` turns are `q.turn` traces tagged `debug`. Replays have their own
trace but skip model calls. Background `q.consolidation` traces use the same session
metadata and record whether the result was applied or the model failed.
Pending trace exports are flushed on application shutdown.

Future tools should use LangChain `@tool` and execute through `.invoke(...)` or a
LangGraph `ToolNode` inside the graph. These emit nested tool spans with names,
arguments, results, duration and errors. For ordinary Python functions called
directly, use LangSmith `@traceable(run_type="tool")`. When dispatching work outside
the graph's execution context, propagate its `RunnableConfig`/trace context so the
tool remains attached to the turn. Tests verify successful and failed tool spans,
inherited session metadata, and replay behavior. Q exposes scoped lookup tools to its
member, episode, observation, intake and handoff agents. Episode lookups are restricted
to the resolved member; the final handoff tool saves the intake note.

Tracing is optional: leave `LANGSMITH_TRACING=false` to keep it off. Traces include
conversation context by default; the assignment uses synthetic data. To omit
inputs and outputs from exports, set `LANGSMITH_HIDE_INPUTS=true` and
`LANGSMITH_HIDE_OUTPUTS=true` (metadata still exports). See the official
[LangChain tracing](https://docs.langchain.com/langsmith/trace-with-langchain) and
[session grouping](https://docs.langchain.com/langsmith/threads) documentation.

## Extension points

The out-of-scope capabilities plug in through `app/extensions.py`: pass an `Extensions` to
`build_graph`. Each is `None` by default and the harness runs without it.

| Capability | Hook | Where it enters the loop |
|---|---|---|
| Custom knowledge base | `knowledge(query) -> passages` | General-question replies get the passages as `reference` |
| Lab report analysis | `lab_results(household_id, member_id) -> results` | A `lab_results` lookup for the intake and handoff agents, offered only when plugged in |
| Telemedicine | `deliver_handoff(note)` | Called after the intake note is saved; a failure fails the turn (503), so Q never claims a handoff it didn't make |
