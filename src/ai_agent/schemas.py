"""Shared streaming event schemas used by services and transport layers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AckEvent(BaseModel):
    type: Literal["ack"] = "ack"
    request_id: str
    session_id: str


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolEndEvent(BaseModel):
    type: Literal["tool_end"] = "tool_end"
    call_id: str
    result: str
    success: bool


class ContentBlockEvent(BaseModel):
    type: Literal["content_block"] = "content_block"
    block: dict[str, Any]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    suggestion: str = ""
    request_id: str = ""


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    request_id: str
    usage: dict[str, int] = Field(default_factory=dict)
