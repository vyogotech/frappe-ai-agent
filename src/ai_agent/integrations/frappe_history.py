"""Best-effort Frappe REST client for chat history: a Frappe outage never aborts the chat."""

from __future__ import annotations

import json
import re
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any
from uuid import uuid4

import httpx
import structlog
from opentelemetry import metrics, trace

logger = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)

# Logs are per call; this counter is what an alert on a sustained Frappe write outage reads.
_meter = metrics.get_meter(__name__)
_history_write_failures = _meter.create_counter(
    name="agent.history.write_failures",
    description="Count of failed AI Chat Session/Message writes to Frappe",
)

_SESSION_URL_PATH = "/api/resource/AI Chat Session"
_MESSAGE_URL_PATH = "/api/resource/AI Chat Message"
_CSRF_URL_PATH = "/app"
_CSRF_HEADER = "X-Frappe-CSRF-Token"
_CSRF_PATTERN = re.compile(r'csrf_token\s*=\s*"([0-9a-fA-F]+)"')
_DEFAULT_TIMEOUT = 10.0


# ponytail: a big table is cut here; the general answer is the tool-output ceiling, P14
_BLOCKS_CHARS = 4000

# Every entry is a live session id, so this cache is a disclosure surface as well as memory.
# ponytail: oldest-out, not least-recently-used; recency would need an OrderedDict and buys
# one desk-page fetch at this size.
_CSRF_CACHE_MAX = 1000


def _with_blocks(text: str, tool_result_json: Any) -> str:
    """The answer as shown, blocks included: a follow-up such as "the first one" points at them."""
    try:
        blocks = json.loads(tool_result_json or "{}").get("blocks") or []
    except (ValueError, AttributeError):
        return text
    if not blocks:
        return text
    shown = json.dumps({"blocks": blocks}, separators=(",", ":"), ensure_ascii=False)
    shown = shown[:_BLOCKS_CHARS]
    return f"{text}\n\n{shown}" if text else shown


class FrappeHistoryClient:
    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Per-sid CSRF token cache. Frappe rotates tokens on re-login; we
        # invalidate an entry whenever a write fails with a CSRF error so
        # the next call picks up the fresh one.
        self._csrf_cache: dict[str, str] = {}
        # One pooled client per instance, made lazily so a process that never writes opens no pool.
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    def _get_client(self) -> httpx.AsyncClient:
        # Raise, not rebuild: a new client here would leak, since aclose() returns once closed.
        if self._closed:
            raise RuntimeError(
                "FrappeHistoryClient is closed; build a new instance for further writes"
            )
        if self._client is None:
            # one client serves every user, so it must never keep the sid a response sets
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the pooled client; idempotent, and the instance is unusable afterwards."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def create_session(
        self,
        *,
        sid: str,
        title: str,
        context_json: str,
    ) -> str | None:
        """Create an AI Chat Session, named here as autoname is "prompt"; None on any failure."""
        url = f"{self._base_url}{_SESSION_URL_PATH}"
        payload = {
            "name": f"chat-{uuid4().hex}",
            "title": title,
            "context_json": context_json,
        }
        return await self._post_and_extract_name(url, payload, sid, "session")

    async def ensure_session(
        self,
        *,
        sid: str,
        name: str,
        title: str,
        context_json: str,
    ) -> str | None:
        """Ensure session `name` exists: a message's Link check fails without its row."""
        url = f"{self._base_url}{_SESSION_URL_PATH}"
        payload = {"name": name, "title": title, "context_json": context_json}
        result = await self._post_and_extract_name(url, payload, sid, "session")
        # If creation returned the name, great. Otherwise assume it failed
        # because the row already exists (idempotent) — return the supplied
        # name so the caller can keep using it.
        return result or name

    async def save_message(
        self,
        *,
        sid: str,
        session: str,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_args_json: str | None = None,
        tool_result_json: str | None = None,
    ) -> str | None:
        """Create an AI Chat Message linked to the given session; None on any failure."""
        url = f"{self._base_url}{_MESSAGE_URL_PATH}"
        payload: dict[str, Any] = {
            "session": session,
            "role": role,
            "content": content,
        }
        if tool_name is not None:
            payload["tool_name"] = tool_name
        if tool_args_json is not None:
            payload["tool_args_json"] = tool_args_json
        if tool_result_json is not None:
            payload["tool_result_json"] = tool_result_json
        return await self._post_and_extract_name(url, payload, sid, "message")

    async def list_messages(
        self,
        *,
        sid: str,
        session: str,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """The last `limit` messages of `session`, oldest first; [] on a Frappe or parse failure."""
        url = f"{self._base_url}/api/method/frappe.client.get_list"
        params = {
            "doctype": "AI Chat Message",
            "fields": json.dumps(["role", "content", "tool_result_json"]),
            "filters": json.dumps([["session", "=", session]]),
            "order_by": "creation desc",
            "limit_page_length": str(max(1, limit)),
        }
        try:
            client = self._get_client()
            resp = await client.get(url, params=params, headers={"Cookie": f"sid={sid}"})
            if resp.status_code != 200:
                logger.warning(
                    "chat_history_list_failed",
                    session=session,
                    status_code=resp.status_code,
                )
                return []
            data = resp.json().get("message") or []
            rows = []
            for r in data:
                content = str(r.get("content") or "")
                if r.get("role") == "assistant" and content.startswith("[error]"):
                    continue  # a failed turn's error text was for the user, not an answer
                if r.get("role") == "assistant":
                    content = _with_blocks(content, r.get("tool_result_json"))
                if r.get("role") in ("user", "assistant") and content:
                    rows.append({"role": str(r["role"]), "content": content})
            rows.reverse()  # oldest-first for LLM context
            return rows
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            logger.warning(
                "chat_history_list_failed",
                session=session,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return []

    # ---------------------------------------------------------------- #
    # internals
    # ---------------------------------------------------------------- #

    async def _fetch_csrf_token(self, sid: str) -> str | None:
        """The CSRF token inlined in the desk page (Frappe sends no header); None on any failure."""
        url = f"{self._base_url}{_CSRF_URL_PATH}"
        # Same closed-check positioning as _post_and_extract_name:
        # outside the try so use-after-close raises cleanly.
        client = self._get_client()
        try:
            # httpx rebuilds a redirect's cookies from the jar, which keeps none, so each hop
            # (Frappe 16 sends /app on to /desk) is followed here with the sid
            response = await client.get(
                url, headers={"Cookie": f"sid={sid}"}, follow_redirects=False
            )
            for _ in range(4):  # five requests in all
                if response.next_request is None:
                    break
                response = await client.get(
                    str(response.next_request.url),
                    headers={"Cookie": f"sid={sid}"},
                    follow_redirects=False,
                )
            response.raise_for_status()
            # A missing, expired or Guest sid ends on /login, whose page has no csrf_token: say so.
            if "/login" in response.url.path:
                logger.warning(
                    "frappe_history_csrf_fetch_unauthenticated",
                    final_url=str(response.url),
                    hint="sid invalid/expired; subsequent writes will 403",
                )
                return None
            match = _CSRF_PATTERN.search(response.text)
            if match is None:
                # Should be rare now that the redirect case is caught above
                # — surface enough detail to diagnose (response size + a
                # short slice) without dumping the whole 400KB desk HTML.
                logger.warning(
                    "frappe csrf token not found in /app response",
                    response_size=len(response.text),
                    response_head=response.text[:200],
                )
                return None
            return match.group(1)
        except httpx.HTTPError as exc:
            logger.warning(
                "frappe_history_csrf_fetch_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    async def _csrf_token_for(self, sid: str) -> str | None:
        cached = self._csrf_cache.get(sid)
        if cached:
            return cached
        fresh = await self._fetch_csrf_token(sid)
        if fresh:
            while len(self._csrf_cache) >= _CSRF_CACHE_MAX:
                del self._csrf_cache[next(iter(self._csrf_cache))]
            self._csrf_cache[sid] = fresh
        return fresh

    def _invalidate_csrf(self, sid: str) -> None:
        self._csrf_cache.pop(sid, None)

    async def _post_and_extract_name(
        self,
        url: str,
        payload: dict[str, Any],
        sid: str,
        kind: str,
    ) -> str | None:
        with _tracer.start_as_current_span("agent.history.write") as span:
            span.set_attribute("kind", kind)
            # Outside the try, which swallows Frappe outages: use-after-close must raise.
            client = self._get_client()

            try:
                csrf_token = await self._csrf_token_for(sid)
                headers: dict[str, str] = {"Cookie": f"sid={sid}"}
                if csrf_token:
                    headers[_CSRF_HEADER] = csrf_token
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 400 and _looks_like_csrf_error(response):
                    # Token probably rotated. Clear cache, refetch, try once.
                    logger.info("frappe_history_csrf_token_rejected_refreshing", kind=kind)
                    self._invalidate_csrf(sid)
                    fresh = await self._csrf_token_for(sid)
                    if fresh:
                        headers[_CSRF_HEADER] = fresh
                        response = await client.post(url, json=payload, headers=headers)

                # 409 on a named session is ensure_session re-posting an existing row: success, not
                # a failed write. Messages are auto-named, so a 409 there is a real failure.
                if response.status_code == 409 and kind == "session" and "name" in payload:
                    logger.info(
                        "frappe_history_session_already_exists",
                        kind=kind,
                        name=payload["name"],
                    )
                    span.set_attribute("status_code", response.status_code)
                    span.set_attribute("idempotent", True)
                    return payload["name"]

                response.raise_for_status()
                span.set_attribute("status_code", response.status_code)
                return response.json()["data"]["name"]
            except Exception as exc:  # noqa: BLE001 - a history write never aborts the answer
                # Transport errors (timeout, DNS, refused) have no response, so status_code is None.
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                logger.warning(
                    "frappe_history_write_failed",
                    kind=kind,
                    session=payload.get("session") or payload.get("name"),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    status_code=status_code,
                )
                _history_write_failures.add(1, {"kind": kind})
                span.set_attribute("failed", True)
                span.set_attribute("error_type", type(exc).__name__)
                if status_code is not None:
                    span.set_attribute("status_code", status_code)
                return None


def _looks_like_csrf_error(response: httpx.Response) -> bool:
    """Whether the body names CSRF; an unreadable body is a no, and any other error propagates."""
    try:
        text = response.text.lower()
    except (UnicodeDecodeError, httpx.ResponseNotRead, httpx.DecodingError):
        return False
    return "csrf" in text
