import typing as t
from pathlib import Path

from dreadnode.core.environment import Environment
from dreadnode.core.environments.containers import ContainerConfig, container

if t.TYPE_CHECKING:
    from dreadnode.core.environments.containers import (
        ContainerContext,
        HealthCheckConfig,
    )


class DockerEnvironment(Environment):
    """Docker container execution environment."""

    def __init__(
        self,
        image: str,
        *,
        name: str | None = None,
        hostname: str | None = None,
        ports: list[int] | None = None,
        env: dict[str, str] | None = None,
        volumes: dict[str | Path, str] | None = None,
        command: list[str] | None = None,
        memory_limit: str | None = None,
        network_name: str | None = None,
        network_isolation: bool = False,
        health_check: "HealthCheckConfig | None" = None,
    ):
        # Import from wherever you put Document 6

        self.image = image
        self.config = ContainerConfig(
            name=name,
            hostname=hostname,
            ports=ports or [],
            env=env or {},
            volumes=volumes or {},
            command=command,
            memory_limit=memory_limit,
            network_name=network_name,
            network_isolation=network_isolation,
            health_check=health_check,
        )

        self._context: ContainerContext | None = None
        self._cm: t.Any = None

    @property
    def container(self) -> "ContainerContext":
        """Access the running container context."""
        if self._context is None:
            raise RuntimeError("Environment not setup - call setup() first")
        return self._context

    async def setup(self) -> dict[str, t.Any]:
        self._cm = container(self.image, self.config)
        self._context = await self._cm.__aenter__()

        return {
            "container_id": self._context.id,
            "hostname": self._context.hostname,
            "ip_address": self._context.ip_address,
            "ports": self._context.ports,
        }

    async def teardown(self) -> None:
        if self._cm:
            await self._cm.__aexit__(None, None, None)
        self._context = None
        self._cm = None

    async def get_state(self) -> dict[str, t.Any]:
        if self._context is None:
            return {"status": "not_running"}
        info = await self._context.container.show()
        return {
            "status": info["State"]["Status"],
            "container_id": self._context.id,
        }

    def tools(self) -> list:
        """Create tools bound to this environment."""
        from dreadnode import tool

        env = self  # Capture for closures

        @tool
        async def run_command(command: str, workdir: str | None = None, timeout: int = 60) -> str:
            """Execute a shell command in the container."""
            exit_code, output = await env.container.run(command, workdir=workdir, timeout=timeout)
            if exit_code != 0:
                return f"Error (exit {exit_code}):\n{output}"
            return output

        @tool
        async def read_file(path: str) -> str:
            """Read a file from the container."""
            exit_code, output = await env.container.run(f"cat {path}")
            if exit_code != 0:
                return f"Error: {output}"
            return output

        @tool
        async def write_file(path: str, content: str) -> str:
            """Write content to a file in the container."""
            import shlex

            escaped = shlex.quote(content)
            cmd = f"mkdir -p $(dirname {path}) && printf '%s' {escaped} > {path}"
            exit_code, output = await env.container.run(cmd)
            if exit_code != 0:
                return f"Error: {output}"
            return f"Wrote {len(content)} bytes to {path}"

        @tool
        async def list_files(path: str = ".", *, recursive: bool = False) -> str:
            """List files in a directory."""
            flag = "-laR" if recursive else "-la"
            exit_code, output = await env.container.run(f"ls {flag} {path}")
            if exit_code != 0:
                return f"Error: {output}"
            return output

        return [run_command, read_file, write_file, list_files]
