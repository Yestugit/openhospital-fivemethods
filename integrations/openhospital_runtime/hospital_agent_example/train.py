"""Local training entrypoint for the baseline agent."""

from __future__ import annotations

import json

from agent import MyDoctorAgent, build_memory
from hospital_agent_sdk import load_config


def main() -> None:
    config = load_config("config.yaml")
    memory = build_memory(config)
    agent = MyDoctorAgent(config=config, memory=memory)
    summary = agent.run_train()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
