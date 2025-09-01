import asyncio
import contextlib
import typing as t
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import aiodocker
import aiodocker.containers
import aiodocker.exceptions
import aiodocker.networks
import aiodocker.types
from loguru import logger

# Helpers


def _parse_memory_limit(limit: str) -> int:
    """Converts a human-readable memory string (e.g., '4g', '512m') to bytes."""
    limit = limit.lower().strip()
    value_str = limit[:-1]
    unit = limit[-1]

    try:
        value = float(value_str)
        if unit == "g":
            return int(value * 1024**3)
        if unit == "m":
            return int(value * 1024**2)
        if unit == "k":
            return int(value * 1024)
        # Assume bytes if no unit
        return int(float(limit))
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid memory limit format: '{limit}'. Use 'g', 'm', 'k' or bytes."
        ) from e


# Config


@dataclass(frozen=True)
class HealthCheckConfig:
    """
    Defines a command-based health check for a container, inspired by Docker Compose.
    """

    command: list[str]
    """The command to run inside the container. Health is determined by exit code 0."""
    interval_seconds: int = 5
    """Seconds to wait between health checks."""
    timeout_seconds: int = 10
    """Seconds to wait for the command to complete before considering it failed."""
    retries: int = 5
    """Number of consecutive failures before marking the container as unhealthy."""
    start_period_seconds: int = 0
    """Grace period for the container to start before checks begin. Failures during this period don't count."""

    async def wait_for_healthy(self, container: aiodocker.containers.DockerContainer) -> None:
        """
        Continuously runs a health check command inside a container until it succeeds
        or the retry limit is exceeded.
        """
        info = await container.show()
        container_name = info["Name"].lstrip("/")

        if self.start_period_seconds > 0:
            logger.debug(
                f"Waiting start period ({self.start_period_seconds}s) for '{container_name}' ..."
            )
            await asyncio.sleep(self.start_period_seconds)

        for retry in range(self.retries):
            try:
                logger.debug(f"Running health check #{retry + 1}: {self.command}")
                exec_instance = await container.exec(self.command)

                async def health_check_task() -> None:
                    async with exec_instance.start() as stream:  # noqa: B023
                        while True:
                            message = await stream.read_out()
                            if message is None:
                                break

                await asyncio.wait_for(health_check_task(), timeout=self.timeout_seconds)
                result = await exec_instance.inspect()

                if result.get("ExitCode") == 0:
                    logger.debug(f"Container '{container_name}' passed health check")
                    return

                logger.debug(
                    f"Health check #{retry + 1} failed with exit code {result.get('ExitCode')}. "
                    f"Retrying in {self.interval_seconds}s ..."
                )

            except asyncio.TimeoutError:
                logger.debug(
                    f"Health check #{retry + 1} timed out after {self.timeout_seconds}s. "
                    f"Retrying in {self.interval_seconds}s ..."
                )
            except Exception as e:
                error = str(e)
                if isinstance(e, aiodocker.exceptions.DockerError):
                    error = e.message
                logger.error(f"Container '{container_name}' health check failed: {error}")
                break

            await asyncio.sleep(self.interval_seconds)

        raise asyncio.TimeoutError(
            f"Container '{container_name}' failed to become healthy after {retry} retries"
        )


@dataclass(frozen=True)
class ContainerConfig:
    name: str | None = None
    """Optional name for the container. If not provided, a random name will be generated."""
    hostname: str | None = None
    """Optional hostname to use for the container (otherwise this will default to the container name)."""
    ports: list[int] = field(default_factory=list)
    """List of ports to expose from the container."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    volumes: dict[str | Path, str] = field(default_factory=dict)
    """Volumes to mount in the container (host path -> container path)."""
    command: list[str] | None = None
    """Command to run in the container (overrides the image's default)."""
    memory_limit: str | None = None
    """Memory limit for the container (e.g., '4g', '512m')."""
    extra_hosts: dict[str, str] = field(default_factory=dict)
    """Additional hostnames to add to the container's /etc/hosts file."""
    network_name: str | None = None
    """Name of the Docker network to connect the container to - will be created if it doesn't exist."""
    network_aliases: list[str] = field(default_factory=list)
    """Aliases for the container in the network."""
    network_isolation: bool = False
    """Whether to isolate the container in its own network."""
    health_check: HealthCheckConfig | None = None
    """An optional, command-based health check to verify container readiness."""

    def merge(self, other: "ContainerConfig | None") -> "ContainerConfig":
        """Merges another config into this one, with 'other' taking precedence."""
        if other is None:
            return self

        return ContainerConfig(
            ports=sorted(set(self.ports) | set(other.ports)),
            env={**self.env, **other.env},
            volumes={**self.volumes, **other.volumes},
            command=other.command or self.command,
            memory_limit=other.memory_limit or self.memory_limit,
            network_name=other.network_name or self.network_name,
            hostname=other.hostname or self.hostname,
            name=other.name or self.name,
            extra_hosts={**self.extra_hosts, **other.extra_hosts},
            network_aliases=list(set(self.network_aliases) | set(other.network_aliases)),
            network_isolation=other.network_isolation or self.network_isolation,
            health_check=other.health_check or self.health_check,
        )


@dataclass(frozen=True)
class ContainerContext:
    """Provides the dynamic runtime context of a running container."""

    id: str
    name: str
    config: ContainerConfig
    hostname: str
    ip_address: str
    ports: dict[int, int]
    network_name: str | None
    container: aiodocker.containers.DockerContainer

    # In the ContainerContext class:

    async def run(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        timeout: int | None = 60,
        shell: str = "/bin/sh",
        privileged: bool = True,
        stream_output: bool = True,
    ) -> tuple[int, str]:
        """
        Executes a command in the container's context, with optional timeout and workdir.

        Args:
            cmd: The command to execute.
            workdir: Optional working directory inside the container.
            timeout: Maximum time to wait for command completion (default 60 seconds) or None for no timeout.
            shell: The shell to use for command execution (default "/bin/sh").
            privileged: Whether to run the command in privileged mode (default True).
            stream_output: If True, display command output in a live Rich panel.

        Returns:
            A tuple of (exit_code, output) where:
            - exit_code: The command's exit code (0 for success, 124 for timeout).
            - output: The command's standard output as a string.
        """
        logger.debug(f"Executing in '{self.name}' ({self.id[:12]}) (timeout: {timeout}s): {cmd}")

        args = [shell, "-c", cmd]
        if timeout is not None:
            args = ["timeout", "-k", "1", "-s", "SIGTERM", str(timeout), *args]

        exec_instance = await self.container.exec(args, privileged=privileged, workdir=workdir)

        output = ""
        with logger.contextualize(prefix=self.name):
            async with exec_instance.start() as stream:
                while True:
                    message = await stream.read_out()
                    if message is None:
                        break
                    chunk = message.data.decode(errors="replace")
                    if stream_output:
                        logger.info(chunk.strip())
                    output += chunk

            inspection = await exec_instance.inspect()
            exit_code = inspection.get("ExitCode", None) or 0
            if exit_code == 124:  # noqa: PLR2004
                logger.warning(f"Command timed out after {timeout}s")

        return exit_code, output


@asynccontextmanager
async def monitor_container(
    container: aiodocker.containers.DockerContainer,
) -> t.AsyncGenerator[None, None]:
    """A context manager to monitor a container and log unexpected exits."""
    shutdown_event = asyncio.Event()

    async def monitor_task_func() -> None:
        try:
            wait_task = asyncio.create_task(container.wait())
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [wait_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

            if wait_task in done and not shutdown_event.is_set():
                info = await container.show()
                exit_code = info["State"]["ExitCode"]
                if exit_code != 0:
                    logs = await container.log(stdout=True, stderr=True)
                    log_str = "".join(logs)
                    container_name = info["Name"].lstrip("/")
                    logger.error(
                        f"Container '{container_name}' ({container.id[:12]}) "
                        f"exited unexpectedly with code {exit_code}:\n{log_str}"
                    )
        except asyncio.CancelledError:
            pass  # Task was cancelled, which is expected on cleanup
        except aiodocker.exceptions.DockerError as e:
            logger.error(f"Error in container monitoring task: {e}")

    monitor_task = asyncio.create_task(monitor_task_func())

    try:
        yield
    finally:
        shutdown_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task


@asynccontextmanager
async def container(  # noqa: PLR0912, PLR0915
    image: str, config: ContainerConfig | None = None, client: aiodocker.Docker | None = None
) -> t.AsyncGenerator[ContainerContext, None]:
    """An async context manager for the full lifecycle of a Docker container."""

    try:
        client = client or aiodocker.Docker()
    except Exception as e:
        raise RuntimeError("Failed to connect to Docker client. Is Docker running?") from e

    container: aiodocker.containers.DockerContainer | None = None
    network: aiodocker.networks.DockerNetwork | None = None
    network_created_by_us = False
    container_name = "unknown"

    config = config or ContainerConfig()

    try:
        # Pull the image if it doesn't exist

        try:
            await client.images.inspect(image)
            logger.info(f"Image '{image}' already exists locally")
        except aiodocker.exceptions.DockerError:
            logger.info(f"Pulling image '{image}'. This may take a moment ...")
            await client.images.pull(image)
            await client.images.inspect(image)
            logger.success(f"Successfully pulled image '{image}'")

        # Setup the network

        if config.network_name:
            try:
                network = await client.networks.get(config.network_name)
                logger.info(f"Using existing network '{config.network_name}'")
            except aiodocker.exceptions.DockerError:
                logger.info(f"Network '{config.network_name}' not found, creating it ...")
                network = await client.networks.create(
                    {"Name": config.network_name, "Driver": "bridge"}
                )
                network_created_by_us = True
                logger.success(f"Created isolated network '{config.network_name}'")

        # Build the config

        extra_hosts = {"host.docker.internal": "host-gateway", **config.extra_hosts}
        host_config: dict[str, t.Any] = {
            "Binds": [f"{Path(h).expanduser().resolve()}:{c}" for h, c in config.volumes.items()],
            "PortBindings": {f"{p}/tcp": [{"HostPort": "0"}] for p in config.ports},
            "ExtraHosts": [f"{k}:{v}" for k, v in extra_hosts.items()],
        }

        if config.memory_limit:
            host_config["Memory"] = str(_parse_memory_limit(config.memory_limit))
            host_config["MemorySwap"] = "-1"  # Disable swap for performance predictability

        create_config: aiodocker.types.JSONObject = {
            "Image": image,
            "Env": [f"{k}={v}" for k, v in config.env.items()],
            "ExposedPorts": {f"{p}/tcp": {} for p in config.ports},
            "HostConfig": host_config,
            "Cmd": config.command,
            "Hostname": config.hostname,
            **({"Entrypoint": ""} if config.command else {}),
        }

        logger.debug(f"Creating container for image '{image}' ...")
        container = await client.containers.create(config=create_config)

        # Connect to network

        if network:
            await network.connect(
                {"Container": container.id, "EndpointConfig": {"Aliases": config.network_aliases}}
            )

        # Start the container

        await container.start()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(container.wait(), timeout=1)

        # Check for non-zero exit code

        info = await container.show()
        if info["State"]["ExitCode"] != 0:
            logs = await container.log(stdout=True, stderr=True)
            log_str = "\n".join(logs)
            raise RuntimeError(f"Container failed to start:\n{log_str}")

        # Gather info

        info = await container.show()
        container_name = info["Name"].lstrip("/")
        logger.info(f"Started container '{container_name}' ({container.id[:12]})")

        mapped_ports = {
            p: int(info["NetworkSettings"]["Ports"][f"{p}/tcp"][0]["HostPort"])
            for p in config.ports
        }
        if mapped_ports:
            logger.info(f"Port mappings: {mapped_ports}")

        container_ip = info["NetworkSettings"]["IPAddress"]

        async with monitor_container(container):
            # Health check

            if config.health_check:
                logger.info(
                    f"Waiting for '{container_name}' ({container.id[:12]}) to be healthy ..."
                )
                await config.health_check.wait_for_healthy(container)
            else:
                logger.debug("No health check configured, assuming container is ready.")

            yield ContainerContext(
                id=container.id,
                name=container_name,
                config=config,
                hostname=config.hostname or container_name,
                ip_address=container_ip,
                ports=mapped_ports,
                network_name=config.network_name,
                container=container,
            )

    finally:
        # Teardown
        if container:
            logger.debug(f"Cleaning up container '{container_name}' ({container.id[:12]}) ...")
            try:
                await container.stop(timeout=1)
                # await container.delete(force=True)
                logger.info(
                    f"Successfully stopped container '{container_name}' ({container.id[:12]})"
                )
            except aiodocker.exceptions.DockerError as e:
                logger.warning(
                    f"Could not remove container '{container_name}' ({container.id[:12]}): {e}"
                )

        if network and network_created_by_us:
            logger.debug(f"Removing network '{network.id}'...")
            await network.delete()
            logger.info(f"Successfully removed network '{network.id}'.")

        await client.close()
