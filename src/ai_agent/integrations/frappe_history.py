"""Frappe REST client for chat history persistence.

Writes AI Chat Session / AI Chat Message DocTypes on behalf of the caller by
forwarding the caller's Frappe sid cookie. Errors are swallowed and logged —
a Frappe outage must NOT abort the conversation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SESSION_URL_PATH = "/api/resource/AI Chat Session"
_MESSAGE_URL_PATH = "/api/resource/AI Chat Message"
_DEFAULT_TIMEOUT = 10.0


class FrappeHistoryClient:
    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def create_session(
        self,
        *,
        sid: str,
        title: str,
        context_json: str,
    ) -> str | None:
        """Create an AI Chat Session owned by the caller.

        Returns the created document's name, or None on any failure.
        """
        url = f"{self._base_url}{_SESSION_URL_PATH}"
        payload = {"title": title, "context_json": context_json}
        return await self._post_and_extract_name(url, payload, sid, "session")

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

    async def _post_and_extract_name(
        self,
        url: str,
        payload: dict[str, Any],
        sid: str,
        kind: str,
    ) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Cookie": f"sid={sid}"},
                )
                response.raise_for_status()
                return response.json()["data"]["name"]
        except Exception as exc:  # noqa: BLE001 — history must never abort the conversation
            logger.warning("frappe history write failed (%s): %s", kind, exc)
            return None
