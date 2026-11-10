"""Agent builder that starts the SDK Flask server."""

from __future__ import annotations

from typing import Optional

from .agent import BasicAgent
from .server import serve_agent

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860


class AgentBuilder:
    def __init__(self, agent: BasicAgent, title: str = "OpenHospital Agent"):
        self.agent = agent
        self.title = title

    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        serve_agent(
            test_handler=self.agent.run_test,
            host=host or DEFAULT_SERVER_HOST,
            port=port if port is not None else DEFAULT_SERVER_PORT,
            title=self.title,
        )
