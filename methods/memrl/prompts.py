"""Prompt text retained from MemRL's OpenAI-compatible provider and updater."""

from __future__ import annotations


def script_prompt(trajectory: str) -> str:
    return (
        f"\n        Analyze the following detailed task trajectory and create a concise,\x20\n"
        "        high-level script that captures the essential steps and decision points.\n"
        "        \x20\n"
        "        The script should be:\n"
        "        1. Generic enough to apply to similar tasks\n"
        "        2. Specific enough to provide useful guidance\n"
        "        3. 3-5 high-level steps maximum\n"
        "        4. Focus on the strategy and key decisions, not detailed actions\n"
        "        \x20\n"
        "        Trajectory:\n"
        f"        {trajectory}\n"
        "        \x20\n"
        "        High-level script:"
    )


def keyword_prompt(text: str, max_keywords: int) -> str:
    return (
        f"\n        Extract up to {max_keywords} key concepts or keywords from the following text.\n"
        "        Focus on the most important nouns, actions, and specific entities.\n"
        "        Return only the keywords separated by commas, nothing else.\n"
        "        \x20\n"
        f"        Text: {text}\n"
        "        \x20\n"
        "        Keywords:"
    )


def reflection_prompt(task_description: str, failed_trajectory: str) -> str:
    return f"""
Task: {task_description}

Failed trajectory:
{failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.
Focus on:
1. Incorrect assumptions
2. Steps to improve
3. What to avoid next time

Provide a brief reflection:
"""
