"""Prompt and parser helpers copied from the ReasoningBank source variant."""
from __future__ import annotations

import re
from dataclasses import dataclass


SUCCESSFUL_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""

FAILED_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""


@dataclass(frozen=True)
class ReasoningNote:
    title: str
    description: str
    content: str


def extraction_prompt(success: bool, *, domain: str = "web navigation") -> str:
    original = SUCCESSFUL_SI if success else FAILED_SI
    return original.replace("an expert in web navigation", f"an expert in {domain}")


def parse_notes(output: str) -> tuple[ReasoningNote, ...]:
    """Split native output like the source runner and parse fields best-effort."""
    if not isinstance(output, str):
        raise TypeError("ReasoningBank extraction must be text")
    notes: list[ReasoningNote] = []
    for block in output.split("\n\n"):
        fields: dict[str, str] = {"Title": "", "Description": "", "Content": block.strip()}
        for name in ("Title", "Description", "Content"):
            match = re.search(rf"(?ms)^##\s+{name}\b[ \t]*(.*?)(?=^##\s|\Z)", block)
            if match and match.group(1).strip():
                fields[name] = match.group(1).strip()
        notes.append(ReasoningNote(fields["Title"], fields["Description"], fields["Content"]))
    return tuple(notes)


def retrieval_query(query: str, *, domain: str = "web navigation") -> str:
    instruction = (f"Given the prior {domain} queries, your task is to analyze a current query's intent "
                   "and select relevant prior queries that could help resolve it.")
    return f"Instruct: {instruction}\nQuery: {query}"
