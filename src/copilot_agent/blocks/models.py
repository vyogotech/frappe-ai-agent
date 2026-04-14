"""Pydantic models for typed content blocks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """Plain markdown text block."""

    type: Literal["text"] = "text"
    content: str


class Dataset(BaseModel):
    """A named series of numeric values."""

    name: str
    values: list[float | int] | list[list[float | int]]


class ChartData(BaseModel):
    """Typed chart data structure."""

    model_config = {"populate_by_name": True, "serialize_by_alias": True}

    labels: list[str]
    datasets: list[Dataset]
    y_labels: list[str] | None = Field(default=None, alias="yLabels")


class ChartOptions(BaseModel):
    """Display options for chart blocks."""

    format: Literal["number", "currency", "percent"] = "number"
    currency: str | None = None
    stacked: bool = False


class ChartBlock(BaseModel):
    """ECharts visualization block."""

    type: Literal["chart"] = "chart"
    chart_type: Literal["bar", "line", "pie", "funnel", "heatmap", "calendar"]
    title: str
    data: ChartData
    options: ChartOptions = ChartOptions()


class Column(BaseModel):
    """Table column definition."""

    key: str
    label: str
    format: Literal["text", "currency", "number", "percent", "date"] = "text"


class DocRoute(BaseModel):
    """Link to an ERPNext document."""

    doctype: str
    name: str


class TableRow(BaseModel):
    """A single table row with optional document link."""

    values: dict[str, Any]
    route: DocRoute | None = None


class TableBlock(BaseModel):
    """Sortable data table block."""

    type: Literal["table"] = "table"
    title: str
    columns: list[Column]
    rows: list[TableRow]


class KPIMetric(BaseModel):
    """A single KPI metric."""

    label: str
    value: float | int | str
    format: Literal["number", "currency", "percent", "text"] = "text"
    trend: Literal["up", "down", "flat"] | None = None
    trend_value: str | None = None


class KPIBlock(BaseModel):
    """Horizontal row of KPI metric cards."""

    type: Literal["kpi"] = "kpi"
    metrics: list[KPIMetric]


class StatusItem(BaseModel):
    """A single status list entry."""

    label: str
    status: str
    color: Literal["green", "red", "yellow", "blue", "gray"]
    route: DocRoute | None = None


class StatusListBlock(BaseModel):
    """Colored status item list."""

    type: Literal["status_list"] = "status_list"
    title: str
    items: list[StatusItem]


ContentBlock = TextBlock | ChartBlock | TableBlock | KPIBlock | StatusListBlock
