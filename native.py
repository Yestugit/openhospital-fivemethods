from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class NativeRunSpec:
    command: tuple[str, ...]
    working_directory: str
    environment: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None


@dataclass(frozen=True)
class NativeRunResult:
    return_code: int
    stdout: str
    stderr: str
    command: tuple[str, ...]
    working_directory: str


class NativeCommandRunner:
    """Runs an upstream experiment without shell interpolation."""

    async def run(self, spec: NativeRunSpec) -> NativeRunResult:
        cwd = Path(spec.working_directory).expanduser().resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"Native working directory does not exist: {cwd}")
        if not spec.command:
            raise ValueError("Native command cannot be empty")
        env = os.environ.copy()
        env.update(spec.environment)
        process = await asyncio.create_subprocess_exec(
            *spec.command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), spec.timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        return NativeRunResult(
            return_code=process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            command=spec.command,
            working_directory=str(cwd),
        )

