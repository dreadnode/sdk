import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dreadnode.agent.tools import Toolset

from .client import HTBClient


@dataclass
class Target:
    """Simple target data structure."""

    name: str
    identifier: str
    type: str
    difficulty: str
    is_active: bool = False
    is_ready: bool = False
    connection_info: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def status(self) -> dict[str, Any]:
        """Get current target status.

        Returns:
            Dict[str, Any]: Dictionary containing status information
        """
        return {
            "is_active": self.is_active,
            "is_ready": self.is_ready,
            "connection_info": self.connection_info,
            **self.metadata,
        }


class HTBPlatform:
    """HTB platform implementation."""

    def __init__(
        self,
        executor: Toolset | None = None,
        *,
        attempts_dir: str = "targets",
        keep_target: bool = False,
        client_debug: bool = False,
    ) -> None:
        """
        Args:
            executor: Executor used for running commands/network.
            attempts_dir: Directory for attempt data & artifacts.
            keep_target: Keep machine running after completion.
            client_debug: Verbose HTBClient logging.
        """
        super().__init__(executor=executor, attempts_dir=attempts_dir)

        self.logger = logging.getLogger(__name__)
        self.keep_target = keep_target

        token = os.getenv("HTB_API_TOKEN")

        self.client = HTBClient(token=token, debug=client_debug)

        # Local VPN artifacts
        self.vpn_dir = Path(__file__).parent / "vpn"
        self.vpn_dir.mkdir(parents=True, exist_ok=True)

    @property
    def platform_name(self) -> str:
        return "HTB"

    # --------------------------- Lifecycle ---------------------------

    def initialize_target(self, target_name: str) -> Target:
        """
        Resolve a machine by name, ensure it's running, and return a ready Target.
        - Reuses currently active machine when it matches.
        - Stops any different active machine before spawning the requested one.
        - Establishes VPN via executor (if provided).
        """
        try:
            machine_info = self.client.get_machine_info(target_name)
            if not machine_info:
                raise RuntimeError(f"Machine {target_name} not found")

            # Already active?
            if machine_info.get("ip"):
                self.logger.info(
                    "Found active machine %s (IP: %s)", target_name, machine_info["ip"]
                )
                target = self._target_from_info(machine_info)
                self._setup_vpn_and_executor(machine_info)
                return target

            # Another machine might be active; stop it.
            active = self.client.get_active_machine()
            if active and (active.get("name") or "").lower() != target_name.lower():
                self.logger.info(
                    "Stopping active machine %s to spawn %s", active.get("name"), target_name
                )
                if not self.client.stop_machine():
                    raise RuntimeError("Failed to stop active machine")

            # Spawn requested machine
            self.logger.info("Spawning machine %s", target_name)
            spawned = self.client.spawn_machine(int(machine_info["id"]), target_name)
            if not spawned:
                raise RuntimeError(f"Failed to spawn machine {target_name}")

            # Wait for IP
            machine_info = self._wait_for_machine_ip(int(machine_info["id"]))

            # Store metadata and writeup
            self._store_machine_metadata(machine_info, spawned)

            # Build target
            target = self._target_from_info(machine_info)

            # Setup VPN + executor
            self._setup_vpn_and_executor(machine_info)

            return target

        except Exception as exc:
            raise RuntimeError(f"Failed to initialize target: {exc!s}") from exc

    def _target_from_info(self, info: dict[str, Any]) -> Target:
        """Create a Target object from HTB machine info dict."""
        t = Target(
            name=info["name"],
            identifier=str(info["id"]),
            type="machine",
            difficulty=info.get("difficulty", "Unknown"),
            metadata={
                "id": info["id"],
                "os": info.get("os", "Unknown"),
                "points": info.get("points", 0),
                "user_owns": info.get("user_owns", 0),
                "root_owns": info.get("root_owns", 0),
                "type": info.get("type", ""),
            },
        )
        t.is_active = True
        t.is_ready = True
        t.connection_info = info.get("ip")
        return t

    def _wait_for_machine_ip(self, machine_id: int, *, timeout: int = 120) -> dict[str, Any]:
        """Poll until active machine shows an IP or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.client.get_active_machine()
            if info and info.get("ip"):
                return info
            time.sleep(5)
        raise RuntimeError(f"Machine {machine_id} did not get IP after {timeout}s")

    def list_targets(self) -> list[dict[str, Any]]:
        """Return union of SP + active + retired machine metadata."""
        machines: list[dict[str, Any]] = []
        machines.extend(self.client.list_starting_point_machines())
        machines.extend(self.client.list_active_machines())
        machines.extend(self.client.list_retired_machines())
        return machines

    def _setup_vpn_and_executor(self, machine_info: dict[str, Any]) -> None:
        """Mount VPN config into the executor and wait until tun0 is up."""
        vpn_server_id = machine_info.get("vpn_server_id")
        if not vpn_server_id:
            raise RuntimeError(
                f"No VPN server ID found for machine {machine_info.get('name')}. "
                "API should include vpn_server_id."
            )

        self.logger.info("Using VPN server %s for %s", vpn_server_id, machine_info.get("name"))
        vpn_config = self.client.get_vpn_config(int(vpn_server_id))

        if not self.executor:
            return

        # Configure executor for VPN
        self.executor.add_capability("NET_ADMIN")
        self.executor.add_device("/dev/net/tun")
        self.executor.add_mount(str(vpn_config), "/tmp/vpn/config.ovpn")

        self.logger.info("Waiting for executor to be ready...")
        if not self.executor.wait_for_ready(timeout=30):
            raise RuntimeError(
                f"Executor '{self.executor.__class__.__name__}' failed to become ready"
            )

        if not self._wait_for_vpn(timeout=60):
            raise RuntimeError("Failed to establish VPN connection")

    def _check_vpn_connected(self) -> bool:
        """Check if tun0 has an inet address inside the executor."""
        if not self.executor:
            return False
        try:
            result = self.executor.execute_command("ip addr show tun0 | grep inet")
            out = result.stdout.strip()
            return bool(out) and 'Device "tun0" does not exist' not in out
        except Exception:  # noqa: BLE001
            return False

    def _wait_for_vpn(self, *, timeout: int = 60) -> bool:
        """Wait until VPN is connected (tun0 up)."""
        self.logger.info("Waiting for VPN connection...")
        start = time.time()
        while time.time() - start < timeout:
            if self._check_vpn_connected():
                self.logger.info("VPN connected successfully")
                return True
            time.sleep(2)
        self.logger.error("VPN failed to connect after %ss", timeout)
        return False

    def _store_machine_metadata(
        self, machine_info: dict[str, Any], spawned_info: dict[str, Any]
    ) -> None:
        """Persist machine metadata and attempt to download official writeup."""
        machine_dir = Path(self.attempts_dir) / machine_info["name"]
        machine_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = machine_dir / "metadata.json"

        if not metadata_file.exists():
            metadata = {
                "id": machine_info["id"],
                "name": machine_info["name"],
                "type": machine_info["type"],
                "difficulty": machine_info.get("difficulty", "Unknown"),
                "vpn_server_id": spawned_info.get("vpn_server_id"),
                "lab_server": spawned_info.get("lab_server"),
            }
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            try:
                self.download_solution(machine_info["name"])
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Could not download writeup: %s", exc)

    def cleanup_target(self, target: Target) -> bool:
        """
        Stop the active machine unless keep_target=True.
        """
        try:
            if not self.keep_target:
                return self.client.stop_machine()
            self.logger.info("Keeping machine %s running as requested", target.name)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to cleanup target: %s", exc)
            return False

    def download_solution(self, target_name: str) -> bool:
        """Try to download the official writeup PDF."""
        try:
            writeup = self.client.get_machine_writeup(target_name)
            if not writeup:
                return False
            machine_dir = Path(self.attempts_dir) / target_name
            machine_dir.mkdir(parents=True, exist_ok=True)
            out = machine_dir / "official_writeup.pdf"
            out.write_bytes(writeup)
            self.logger.info("Downloaded writeup to %s", out)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to download writeup: %s", exc)
            return False

    # --------------------------- Flag validation & prompts ---------------------------

    def validate_flag(self, flag: str, target: Target | None = None) -> bool:
        """
        Submit a flag to HTB API for validation. Requires target metadata.id.
        """
        if not target:
            self.logger.warning("No target provided to validate flag against")
            return False

        machine_id = target.metadata.get("id") if target.metadata else None
        if not machine_id:
            self.logger.warning("No machine ID found in target metadata")
            return False

        try:
            result = self.client.submit_flag(int(machine_id), flag)
            if result.get("success", False):
                self.logger.info("Flag validation successful: %s", result.get("message"))
                return True
            self.logger.warning("Flag validation failed: %s", result.get("message"))
            return False
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error validating flag: %s", exc)
            return False

    def get_platform_prompt(self, target: Target, template_vars: dict[str, Any]) -> str:  # type: ignore[override]
        """
        Render platform-specific prompt for HTB targets.
        Uses 'starting_point_instructions.yaml' for SP; otherwise 'machine_instructions.yaml'.
        """
        import yaml
        from jinja2 import Template

        is_sp = bool(
            getattr(target, "metadata", {}) and target.metadata.get("type") == "Starting Point"
        )

        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        base = prompts_dir / self.platform_name.lower()
        path = base / ("starting_point_instructions.yaml" if is_sp else "machine_instructions.yaml")

        if not path.exists():
            raise FileNotFoundError(f"Platform-specific prompt file not found: {path}")

        self.logger.debug("Reading platform prompt: %s", path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Template(data["target_prompt"]).render(**template_vars)

    def _check_target_readiness(
        self, connection_info: str, *, max_retries: int = 10, retry_delay: int = 15
    ) -> bool:
        """Ping the target via executor until reachable or timeout."""
        if not self.executor:
            raise RuntimeError("Executor not provided, cannot check target readiness")

        self.logger.info("Checking if target %s is ready...", connection_info)
        for _ in range(max_retries):
            try:
                res = self.executor.execute_command(f"ping -c 1 {connection_info}")
                if res.exit_code == 0:
                    self.logger.info("Target %s is responding to ping", connection_info)
                    return True
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Ping attempt failed: %s", exc)
            time.sleep(retry_delay)
        return False
