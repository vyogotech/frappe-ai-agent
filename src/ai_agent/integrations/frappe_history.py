"""Frappe REST client for chat history persistence.

Writes AI Chat Session / AI Chat Message DocTypes on behalf of the caller by
forwarding the caller's Frappe sid cookie. Errors are swallowed and logged —
a Frappe outage must NOT abort the conversation.

CSRF handling: Frappe protects state-changing REST endpoints with a CSRF
token. Frappe v17 embeds the token as a JS variable inside the rendered
`/app` HTML page (`csrf_token = "<hex>"`), NOT as a response header.
We GET `/app`, regex out the token, cache it per sid, and attach it
as `X-Frappe-CSRF-Token` on every write. If a write fails with a CSRF
error we invalidate the cache so the next call re-fetches.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

_SESSION_URL_PATH = "/api/resource/AI Chat Session"
_MESSAGE_URL_PATH = "/api/resource/AI Chat Message"
_CSRF_URL_PATH = "/app"
_CSRF_HEADER = "X-Frappe-CSRF-Token"
_CSRF_PATTERN = re.compile(r'csrf_token\s*=\s*"([0-9a-fA-F]+)"')
_DEFAULT_TIMEOUT = 10.0


class FrappeHistoryClient:
    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # Per-sid CSRF token cache. Frappe rotates tokens on re-login; we
        # invalidate an entry whenever a write fails with a CSRF error so
        # the next call picks up the fresh one.
        self._csrf_cache: dict[str, str] = {}
        # Long-lived AsyncClient reused across all calls from this
        # instance. Opening a fresh client per write paid a new TCP
        # connection setup (plus TLS handshake when behind HTTPS) per
        # 3-4 calls per chat turn. The single client gets a connection
        # pool keyed by host and reuses it. Lazy-init so a Settings()
        # default doesn't force a connection pool at config-load time
        # for processes that never touch Frappe (CLI tools, tests).
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying AsyncClient. Idempotent.

        Called from the FastAPI lifespan teardown; safe to call multiple
        times (lifespan exit may run after a context manager already
        cleaned up). After aclose the instance is unusable — a fresh
        instance must be built for any further writes.
        """
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
        """Create an AI Chat Session owned by the caller.

        Returns the created document's name, or None on any failure.

        The AI Chat Session DocType is declared ``autoname: "prompt"`` —
        Frappe requires callers to supply the row's primary key. Generate a
        UUID-based name here so the resulting id is opaque to the user and
        collision-free across concurrent first turns.
        """
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
        """Ensure an AI Chat Session with this exact ``name`` exists.

        Used when the caller supplies a conversation id (e.g. forwarded by
        Frappe from the browser) — subsequent message writes' Link validation
        would 417 against a missing parent row. We attempt to create with the
        explicit ``name`` field; a duplicate-name conflict means the session
        is already there from an earlier turn, which is success.
        """
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
        """Create an AI Chat Message linked to the given session.

        Returns the created document's name, or None on any failure.
        """
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

    # ---------------------------------------------------------------- #
    # internals
    # ---------------------------------------------------------------- #

    async def _fetch_csrf_token(self, sid: str) -> str | None:
        """GET /app and extract the CSRF token from the rendered HTML.

        Frappe v17 embeds the token as `csrf_token = "<hex>"` inline in
        the desk page JavaScript. Following redirects lets us land on
        the real desk page even if /app redirects.

        Returns None on any failure so callers can still attempt the write
        (Frappe will return a clear 400 CSRFTokenError we log downstream).
        """
        url = f"{self._base_url}{_CSRF_URL_PATH}"
        try:
            client = self._get_client()
            response = await client.get(
                url,
                headers={"Cookie": f"sid={sid}"},
            )
            response.raise_for_status()
            match = _CSRF_PATTERN.search(response.text)
            if match is None:
                logger.warning("frappe csrf token not found in /app response")
                return None
            return match.group(1)
        except Exception as exc:
            logger.warning("frappe csrf fetch failed: %s", exc)
            return None

    async def _csrf_token_for(self, sid: str) -> str | None:
        cached = self._csrf_cache.get(sid)
        if cached:
            return cached
        fresh = await self._fetch_csrf_token(sid)
        if fresh:
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
        csrf_token = await self._csrf_token_for(sid)
        headers: dict[str, str] = {"Cookie": f"sid={sid}"}
        if csrf_token:
            headers[_CSRF_HEADER] = csrf_token

        try:
            client = self._get_client()
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code == 400 and _looks_like_csrf_error(response):
                # Token probably rotated. Clear cache, refetch, try once.
                logger.info("frappe csrf token rejected, refreshing once")
                self._invalidate_csrf(sid)
                fresh = await self._csrf_token_for(sid)
                if fresh:
                    headers[_CSRF_HEADER] = fresh
                    response = await client.post(url, json=payload, headers=headers)

            response.raise_for_status()
            return response.json()["data"]["name"]
        except Exception as exc:
            logger.warning("frappe history write failed (%s): %s", kind, exc)
            return None


def _looks_like_csrf_error(response: httpx.Response) -> bool:
    """Best-effort check for a Frappe CSRFTokenError response body."""
    try:
        text = response.text.lower()
    except Exception:
        return False
    return "csrf" in text
