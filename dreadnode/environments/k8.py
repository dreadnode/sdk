import typing as t

from dreadnode.core.environment import Environment


class KubernetesSandboxEnvironment(Environment):
    """
        Kubernetes Agent Sandbox environment.

        Uses the Kubernetes-native Agent Sandbox for:
        - gVisor-based kernel isolation
        - Ephemeral, secure sandboxes
        - Kubernetes-native orchestration
        - Scalable parallel execution

        Args:
            template: Name of the SandboxTemplate to use
            namespace: Kubernetes namespace
            kubeconfig: Path to kubeconfig file (defaults to ~/.kube/config)
            timeout: Sandbox timeout in seconds
            router_address: Address of the sandbox router (if not using port-forward)

        Example:
    ```python
            from dreadnode.core.agents.environment import KubernetesAgentSandboxEnvironment

            agent = Agent(
                name="secure-executor",
                environment=KubernetesSandbox(
                    template="python-sandbox",
                    namespace="ai-sandboxes",
                ),
                tools=[run_command],
            )
    ```
    """

    def __init__(
        self,
        template: str = "default",
        *,
        namespace: str = "default",
        kubeconfig: str | None = None,
        timeout: int = 300,
        router_address: str | None = None,
    ):
        self.template = template
        self.namespace = namespace
        self.kubeconfig = kubeconfig
        self.timeout = timeout
        self.router_address = router_address

        self._sandbox: t.Any = None
        self._client: t.Any = None

    @property
    def sandbox(self) -> t.Any:
        """Access the running sandbox."""
        if self._sandbox is None:
            raise RuntimeError("Environment not setup - call setup() first")
        return self._sandbox

    async def setup(self) -> dict[str, t.Any]:
        """Create the Kubernetes sandbox."""
        try:
            from agentic_sandbox import Sandbox, SandboxConfig
        except ImportError:
            raise ImportError(
                "agentic-sandbox is required for KubernetesAgentSandboxEnvironment. "
                "See: https://github.com/kubernetes-sigs/agent-sandbox"
            )

        config = SandboxConfig(
            template=self.template,
            namespace=self.namespace,
            kubeconfig=self.kubeconfig,
            timeout=self.timeout,
        )
        if self.router_address:
            config.router_address = self.router_address

        self._sandbox = Sandbox(config)
        await self._sandbox.create()
        await self._sandbox.wait_ready()

        return {
            "sandbox_name": self._sandbox.name,
            "namespace": self.namespace,
            "template": self.template,
        }

    async def teardown(self) -> None:
        """Delete the Kubernetes sandbox."""
        if self._sandbox is not None:
            await self._sandbox.delete()
            self._sandbox = None

    async def reset(self) -> dict[str, t.Any]:
        """Reset by recreating the sandbox."""
        await self.teardown()
        return await self.setup()

    async def get_state(self) -> dict[str, t.Any]:
        """Get current sandbox state."""
        if self._sandbox is None:
            return {"status": "not_running"}

        status = await self._sandbox.get_status()
        return {
            "status": status,
            "sandbox_name": self._sandbox.name,
            "namespace": self.namespace,
        }

    def tools(self) -> list:
        """Create tools for Kubernetes sandbox."""
        from dreadnode import tool

        env = self

        @tool
        async def run_command(command: str, timeout: int = 60) -> str:
            """
            Execute a shell command in the Kubernetes sandbox.
            The sandbox provides gVisor isolation for security.
            """
            result = await env.sandbox.run(command, timeout=timeout)

            output = result.stdout or ""
            if result.stderr:
                output += f"\nstderr: {result.stderr}"
            if result.exit_code != 0:
                output = f"Exit code {result.exit_code}\n{output}"

            return output or "(no output)"

        @tool
        async def execute_python(code: str) -> str:
            """Execute Python code in the sandbox."""
            # Write code to temp file and execute
            import uuid

            filename = f"/tmp/code_{uuid.uuid4().hex[:8]}.py"

            await env.sandbox.write_file(filename, code)
            result = await env.sandbox.run(f"python3 {filename}")

            output = result.stdout or ""
            if result.stderr:
                output += f"\nstderr: {result.stderr}"
            if result.exit_code != 0:
                output = f"Error (exit {result.exit_code}):\n{output}"

            return output or "(no output)"

        return [run_command, execute_python]
