"""An optional usage gate: may this site's user send another message?

Off unless `usage_check_url` is configured, so an agent run on its own behaves
exactly as before. An operator serving many sites from one agent can point it
at their own usage service; that service owns quotas and counting, the agent
only asks before answering.

It lives in the agent, not in the Frappe site, because the agent is the shared
service: a check inside a site can be bypassed by anyone able to call the
agent from that site, a check here cannot.

An allowed answer may also carry `llm`: the base URL, key and user to call the
model with for this site's user, for an operator whose sites use different
model accounts. Without it the agent uses its own settings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import structlog

from ai_agent.config import Settings

logger = structlog.get_logger(__name__)

NOT_AVAILABLE = "The AI Assistant is not available on this site right now."
UNAVAILABLE = "The AI Assistant is temporarily unavailable. Try again in a minute."


@dataclass(frozen=True)
class LLMCredentials:
    base_url: str
    api_key: str
    user: str | None = None


@dataclass(frozen=True)
class UsageAnswer:
    refusal: str | None = None
    credentials: LLMCredentials | None = None


# (site, user) -> (expires at, credentials). Per process; a miss only means one more ask.
_kept: dict[tuple[str, str], tuple[float, LLMCredentials]] = {}
_KEPT_MAX = 10_000


def _key(site: str | None, user: str | None) -> tuple[str, str]:
    return (str(site or "").lower(), str(user or "").lower())


def _keep(
    settings: Settings, site: str | None, user: str | None, credentials: LLMCredentials
) -> None:
    if len(_kept) >= _KEPT_MAX:
        now = time.monotonic()
        for k in [k for k, (expires, _) in _kept.items() if expires <= now] or list(_kept)[
            : _KEPT_MAX // 10
        ]:
            _kept.pop(k, None)
    _kept[_key(site, user)] = (time.monotonic() + settings.usage_credentials_ttl_s, credentials)


def kept_credentials(site: str | None, user: str | None) -> LLMCredentials | None:
    """Credentials an allowed answer gave this site's user within the TTL, or None."""
    entry = _kept.get(_key(site, user))
    if not entry:
        return None
    expires, credentials = entry
    if expires <= time.monotonic():
        _kept.pop(_key(site, user), None)
        return None
    return credentials


def _parse(answer: dict) -> LLMCredentials | None:
    llm = answer.get("llm")
    if not isinstance(llm, dict) or not llm.get("base_url") or not llm.get("api_key"):
        return None
    return LLMCredentials(
        base_url=str(llm["base_url"]).rstrip("/"),
        api_key=str(llm["api_key"]),
        user=llm.get("user") or None,
    )


def without_answer(settings: Settings, site: str | None, user: str | None) -> UsageAnswer:
    """No usable answer from the service: kept credentials, else what the settings allow."""
    kept = kept_credentials(site, user)
    if kept:
        return UsageAnswer(credentials=kept)
    return UsageAnswer(refusal=UNAVAILABLE if settings.llm_credentials_required else None)


async def check(settings: Settings, site: str | None, user: str | None) -> UsageAnswer:
    """Ask the usage service about one typed message.

    Unreachable or failing (5xx) service: go ahead with credentials kept from
    an earlier answer, or as before when none are kept and none are required, so its outage does not
    take every site's assistant down. A 4xx (the agent's token refused, or a
    site the service does not know) is a refusal: a gate that opens when it
    cannot identify the caller is not a gate.
    """
    url = settings.usage_check_url.strip()
    if not url:
        return UsageAnswer()
    headers = {"Accept": "application/json"}
    if settings.usage_check_token:
        # Its own header, not Authorization, so a service that authenticates
        # users by Authorization cannot mistake it for a user's credential.
        headers["X-Usage-Token"] = settings.usage_check_token
    try:
        async with httpx.AsyncClient(timeout=settings.usage_check_timeout_s) as client:
            response = await client.post(url, json={"site": site, "user": user}, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("usage_check.unreachable", site=site, error=str(exc))
        return without_answer(settings, site, user)
    if response.status_code >= 500:
        logger.warning("usage_check.failed", site=site, status=response.status_code)
        return without_answer(settings, site, user)
    if response.status_code >= 400:
        logger.warning("usage_check.refused_caller", site=site, status=response.status_code)
        return UsageAnswer(refusal=NOT_AVAILABLE)
    try:
        answer = (response.json() or {}).get("message") or {}
    except ValueError:
        logger.warning("usage_check.bad_answer", site=site)
        return without_answer(settings, site, user)
    if not answer.get("allowed"):
        return UsageAnswer(refusal=str(answer.get("reason") or NOT_AVAILABLE))
    credentials = _parse(answer)
    if credentials:
        _keep(settings, site, user, credentials)
        return UsageAnswer(credentials=credentials)
    if settings.llm_credentials_required:
        logger.warning("usage_check.no_credentials", site=site)
        return UsageAnswer(refusal=UNAVAILABLE)
    return UsageAnswer()


async def refusal_reason(settings: Settings, site: str | None, user: str | None) -> str | None:
    """The usage service's reason to refuse this message, or None to go ahead."""
    return (await check(settings, site, user)).refusal
