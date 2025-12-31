import typing as t
from pathlib import Path

from dreadnode.core.environment import Environment

if t.TYPE_CHECKING:
    from dreadnode.core.environments.kernel import PythonKernel


class PythonKernelEnvironment(Environment):
    """Jupyter kernel execution environment."""

    def __init__(
        self,
        image: str = "jupyter/datascience-notebook:latest",
        *,
        memory_limit: str = "4g",
        kernel_name: str = "python3",
        packages: list[str] | None = None,
        work_dir: Path | str | None = None,
        volumes: list[str] | None = None,
    ):
        self.packages = packages or []
        self._kernel = PythonKernel(
            image=image,
            memory_limit=memory_limit,
            kernel_name=kernel_name,
            work_dir=work_dir,
            volumes=volumes,
        )
        self._is_setup = False

    @property
    def kernel(self) -> "PythonKernel":
        """Access the running kernel."""
        if not self._is_setup:
            raise RuntimeError("Environment not setup - call setup() first")
        return self._kernel

    async def setup(self) -> dict[str, t.Any]:
        await self._kernel.init()

        if self.packages:
            pkg_str = " ".join(self.packages)
            result = await self._kernel.execute(f"!pip install -q {pkg_str}", timeout=120)
            if result.error:
                raise RuntimeError(f"Failed to install packages: {result.error}")

        self._is_setup = True

        return {
            "kernel_id": self._kernel._kernel_id,
            "work_dir": str(self._kernel._work_dir),
            "packages": self.packages,
        }

    async def teardown(self) -> None:
        await self._kernel.shutdown()
        self._is_setup = False

    async def reset(self) -> dict[str, t.Any]:
        """Restart kernel (faster than full teardown)."""
        if self._is_setup:
            await self._kernel.restart()
            if self.packages:
                pkg_str = " ".join(self.packages)
                await self._kernel.execute(f"!pip install -q {pkg_str}", timeout=120)
            return await self.get_state()
        return await super().reset()

    async def get_state(self) -> dict[str, t.Any]:
        if not self._is_setup:
            return {"status": "not_running"}
        try:
            state = await self._kernel.get_kernel_state()
            return {"status": state, "kernel_id": self._kernel._kernel_id}
        except Exception:
            return {"status": "error"}

    def tools(self) -> list:
        """Create tools bound to this environment."""
        from dreadnode import tool

        env = self  # Capture for closures

        @tool
        async def execute_python(code: str, timeout: int = 30) -> str:
            """
            Execute Python code in the Jupyter kernel.
            State persists between calls (variables, imports, etc.).
            """
            result = await env.kernel.execute(code, timeout=timeout)
            output = result.to_str()
            if result.error:
                return f"Error:\n{result.error}\n\nOutput:\n{output}"
            return output or "(no output)"

        @tool
        async def restart_kernel() -> str:
            """Restart the Python kernel, clearing all state."""
            await env.kernel.restart()
            return "Kernel restarted. All state cleared."

        return [execute_python, restart_kernel]
