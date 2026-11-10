"""Verbatim text templates from MemQ's bundled OpenAI provider and updater."""


def script_prompt(trajectory: str) -> str:
    return f"""
        Analyze the following detailed task trajectory and create a concise,
        high-level script that captures the essential steps and decision points.

        The script should be:
        1. Generic enough to apply to similar tasks
        2. Specific enough to provide useful guidance
        3. 3-5 high-level steps maximum
        4. Focus on the strategy and key decisions, not detailed actions

        Trajectory:
        {trajectory}

        High-level script:"""


def keyword_prompt(text: str, max_keywords: int = 8) -> str:
    return f"""
        Extract up to {max_keywords} key concepts or keywords from the following text.
        Focus on the most important nouns, actions, and specific entities.
        Return only the keywords separated by commas, nothing else.
        
        Text: {text}
        
        Keywords:"""


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
