"""Sanitize and truncate oversized content blocks."""

from __future__ import annotations

from typing import TypeVar

from ai_agent.blocks.models import (
    ChartBlock,
    ContentBlock,
    KPIBlock,
    StatusListBlock,
    TableBlock,
)

MAX_TABLE_ROWS = 100
MAX_CHART_DATAPOINTS = 500
MAX_KPI_METRICS = 8
MAX_STATUS_ITEMS = 50

# Generic in the input type so callers that pass a TableBlock get a TableBlock
# back (not the wider ContentBlock union). This lets pyright narrow attribute
# access on the result without a redundant isinstance check.
B = TypeVar("B", bound=ContentBlock)


def validate_block(block: B) -> B:
    """Apply truncation limits to a content block."""
    if isinstance(block, TableBlock) and len(block.rows) > MAX_TABLE_ROWS:
        return block.model_copy(update={"rows": block.rows[:MAX_TABLE_ROWS]})
    elif isinstance(block, ChartBlock):
        truncated_datasets = []
        for ds in block.data.datasets:
            if len(ds.values) > MAX_CHART_DATAPOINTS:
                truncated_datasets.append(
                    ds.model_copy(update={"values": ds.values[:MAX_CHART_DATAPOINTS]})
                )
            else:
                truncated_datasets.append(ds)
        if truncated_datasets != block.data.datasets:
            new_data = block.data.model_copy(update={"datasets": truncated_datasets})
            return block.model_copy(update={"data": new_data})
    elif isinstance(block, KPIBlock) and len(block.metrics) > MAX_KPI_METRICS:
        return block.model_copy(update={"metrics": block.metrics[:MAX_KPI_METRICS]})
    elif isinstance(block, StatusListBlock) and len(block.items) > MAX_STATUS_ITEMS:
        return block.model_copy(update={"items": block.items[:MAX_STATUS_ITEMS]})
    return block
