"""Harbor agent for Command Code (`cmd`), run headless inside the task container.

Harbor imports this with the harbor tool's own Python, so it must not import anything from
smol_ladder or its heavy dependencies.

    harbor run -p <tasks> --agent-import-path harbor_ext.cmd_agent:CommandCode \
        -m stealth/space-bunny-alpha
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

AUTH = Path.home() / ".commandcode" / "auth.json"


class CommandCode(BaseInstalledAgent):
    @staticmethod
    @override
    def name() -> str:
        return "command-code"

    @override
    def get_version_command(self) -> str | None:
        return "command-code --version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Node and command-code come from the smol-ladder base image; only the login is
        # copied in per trial, never baked into an image.
        await environment.upload_file(AUTH, "/tmp/cc-auth.json")
        await self.exec_as_agent(
            environment,
            command="mkdir -p ~/.commandcode && mv /tmp/cc-auth.json ~/.commandcode/auth.json "
            "&& chmod 600 ~/.commandcode/auth.json && command -v command-code",
        )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        if not self.model_name:
            raise ValueError("pass -m, e.g. -m stealth/space-bunny-alpha")
        await self.exec_as_agent(
            environment,
            command=(
                f"command-code -p {shlex.quote(instruction)} -m {shlex.quote(self.model_name)} "
                "--yolo -t --skip-onboarding --no-session --max-turns 40 --output-format json "
                "</dev/null > /logs/agent/cmd.jsonl 2> /logs/agent/cmd.stderr"
            ),
        )
