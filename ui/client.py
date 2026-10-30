"""HTTP boundary for the Streamlit app; the API owns the agent and database."""

import httpx


class ChatAPIError(RuntimeError):
    pass


def request(api_url: str, method: str, path: str, payload: dict | None = None):
    try:
        response = httpx.request(
            method, f"{api_url.rstrip('/')}{path}", json=payload,
            timeout=httpx.Timeout(120, connect=5),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # A close after a failed first message can legitimately find no active session.
        if path.endswith("/close") and exc.response.status_code == 404:
            return {"status": "closed"}
        messages = {
            404: "This session or profile is unavailable. Check the account and restart the API.",
            409: "This message conflicts with an earlier turn. Start a new chat.",
            422: "The message was not accepted. Messages must contain 1–2,000 characters.",
            502: "Q could not generate a reply. Please retry your message.",
            503: "Q is temporarily unavailable. Please retry shortly.",
        }
        status = exc.response.status_code
        message = messages.get(status, "Q could not complete this request. Please retry.")
        if status in (502, 503):  # a failed turn or handoff: the API's text carries the safety advice
            message = _detail(exc.response) or message
        raise ChatAPIError(message) from exc
    except httpx.RequestError as exc:
        raise ChatAPIError("Cannot reach Q. Check that the API is running, then retry.") from exc
    return response.json()


def _detail(response: httpx.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return detail if isinstance(detail, str) else None
