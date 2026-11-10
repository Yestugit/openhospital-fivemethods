"""Read memory output and its declared sources without executing an Agent."""
from copy import deepcopy

from mematlas.core.models import MemoryContribution, content_text


def contribution_text(contribution: MemoryContribution) -> str:
    rows = contribution.context or contribution.items
    return "\n\n".join(content_text(row["content"]) for row in rows if content_text(row["content"]))


def applied_sources(contributions):
    """Extract sources from contributions the caller actually applied."""
    output = []
    for contribution in contributions:
        indices = list(dict.fromkeys(i for row in contribution.context for i in row["source_item_indices"])) if contribution.context else range(len(contribution.items))
        output.extend(deepcopy(contribution.items[i]) for i in indices)
    return tuple(output)
