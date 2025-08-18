from __future__ import annotations

"""
HTB API client (modernized).

- httpx (HTTP/2, timeouts) instead of requests
- Strong typing + docstrings
- Safer pagination, backoff, and caching
- Clean CLI wiring to pass --debug into client
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import httpx

# Toggle extra wire logs if needed (kept as a constant switch)
REQUEST_DEBUG: Final[bool] = False

BASE_URL: Final[str] = "https://labs.hackthebox.com/api/v4"
APP_URL: Final[str] = "https://app.hackthebox.com/api/v4"
DEFAULT_UA: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class HTBClient:
    """Typed client for Hack The Box APIs."""

    def __init__(
        self, token: str, *, use_cache: bool = True, debug: bool = False, timeout_s: float = 20.0
    ) -> None:
        """
        Args:
            token: HTB API token.
            use_cache: Whether to use on-disk caches.
            debug: Enable verbose logs.
            timeout_s: HTTP timeout.
        """
        self.base_url: str = BASE_URL
        self.app_url: str = APP_URL
        self.token: str = token
        self.use_cache: bool = use_cache
        self.debug: bool = debug

        self.logger = logging.getLogger(__name__)
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_UA,
                "Origin": "https://app.hackthebox.com",
                "Referer": "https://app.hackthebox.com/",
            },
            timeout=httpx.Timeout(timeout_s),
            http2=True,
            follow_redirects=False,
        )

        # Cache & VPN directories adjacent to this file
        root = Path(__file__).parent
        self.cache_dir: Path = root / ".apicache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.vpn_dir: Path = root / "vpn"
        self.vpn_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- HTTP helpers ---------------------------

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Unified request with optional debug logging."""
        resp = self._client.request(method, url, **kwargs)
        if REQUEST_DEBUG or self.debug:
            self._log_response(method, url, resp)
        return resp

    def _log_response(self, method: str, url: str, resp: httpx.Response) -> None:
        self.logger.debug("%s %s -> %s", method, url, resp.status_code)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                self.logger.debug("Response JSON: %s", resp.json())
            except Exception as exc:  # noqa: BLE001
                self.logger.debug("Failed to decode JSON: %s", exc)
                self.logger.debug("Raw: %s", resp.text[:1000])
        else:
            self.logger.debug("Response content-type: %s", ctype)

    def _paginated_request(self, url: str, page: int = 1) -> dict[str, Any] | None:
        """
        Handle paginated GET with rate limiting backoff.

        Returns parsed JSON or None on failure.
        """
        max_retries = 5
        base_delay = 1.0  # seconds
        retry = 0

        while retry < max_retries:
            resp = self._request("GET", url, params={"page": page})
            if resp.status_code == 429:
                retry += 1
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = base_delay * (2**retry)
                if self.debug:
                    self.logger.debug("Rate limited. Retrying in %.1fs...", delay)
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    if self.debug:
                        self.logger.debug("Failed to parse JSON from %s", url)
                    return None

            if self.debug:
                self.logger.debug("Paginated request failed: %s", resp.status_code)
            return None

        self.logger.error("Max retries reached due to rate limiting for %s", url)
        return None

    # --------------------------- Cache helpers ---------------------------

    def _load_cache(self, category: str) -> dict[str, Any] | None:
        if not self.use_cache:
            return None
        file = self.cache_dir / f"{category}.json"
        if not file.exists():
            return None
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.logger.debug("Invalid JSON in %s cache", category)
            return None

    def _save_cache(self, category: str, data: dict[str, Any]) -> None:
        file = self.cache_dir / f"{category}.json"
        file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.logger.debug("Updated %s cache", category)

    # --------------------------- Active machine ---------------------------

    def _check_active_machine(self) -> dict[str, Any] | None:
        """Return active machine info (or None). Retries spawn in-progress."""
        resp = self._request("GET", f"{self.base_url}/machine/active")
        if resp.status_code != 200:
            return None

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return None

        info = data.get("info")
        if not info:
            return None

        # If spawning, hand off to waiter
        if info.get("isSpawning", False):
            return self._wait_for_active_machine(info.get("name", ""), int(info.get("id", 0)))

        return {
            "id": info.get("id"),
            "name": info.get("name", "Unknown"),
            "type": info.get("type"),
            "ip": info.get("ip"),
            "isSpawning": info.get("isSpawning", False),
            "vpn_server_id": info.get("vpn_server_id"),
            "lab_server": info.get("lab_server"),
            "tier_id": info.get("tier_id"),
        }

    def get_active_machine(self) -> dict[str, Any] | None:
        """Public accessor for the active machine."""
        return self._check_active_machine()

    def stop_machine(self) -> bool:
        """Stop currently active machine."""
        resp = self._request("POST", f"{self.base_url}/vm/terminate", json={"machine_id": None})
        return resp.status_code in (200, 201)

    def spawn_machine(self, machine_id: int, machine_name: str) -> dict[str, Any] | None:
        """Spawn a machine and wait until ready."""
        resp = self._request("POST", f"{self.base_url}/vm/spawn", json={"machine_id": machine_id})
        if resp.status_code not in (200, 201):
            self.logger.error("Failed to spawn machine: %s", resp.text)
            return None
        return self._wait_for_active_machine(machine_name, machine_id)

    def _wait_for_active_machine(
        self, machine_name: str, machine_id: int, *, timeout: int = 180
    ) -> dict[str, Any] | None:
        """
        Poll the 'active' endpoint until the named machine has an IP (with bounded retries).
        Includes a restart loop with increasing timeouts if the VM stalls.
        """
        self.logger.info(
            "Waiting for machine %s to become active (can take ~1 minute)...", machine_name
        )
        start = time.time()
        retries_nonjson = 3
        attempt = 1
        max_attempts = 5
        backoff_factor = 1.5
        current_timeout = timeout

        while attempt <= max_attempts:
            try:
                resp = self._request("GET", f"{self.base_url}/machine/active")
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except json.JSONDecodeError:
                        if retries_nonjson > 0:
                            self.logger.warning("Non-JSON response, retrying...")
                            retries_nonjson -= 1
                            time.sleep(4)
                            continue
                        if attempt < max_attempts:
                            attempt += 1
                            retries_nonjson = 3
                            continue
                        return None

                    active = data.get("info")
                    if active and active.get("name", "").lower() == machine_name.lower():
                        if active.get("isSpawning", False) or not active.get("ip"):
                            elapsed = time.time() - start
                            if elapsed > current_timeout:
                                self.logger.warning(
                                    "Timeout after %.0fs (attempt %d/%d). Restarting...",
                                    elapsed,
                                    attempt,
                                    max_attempts,
                                )
                                if attempt < max_attempts:
                                    if not self.stop_machine():
                                        self.logger.error("Failed to stop stalled machine")
                                    time.sleep(5)
                                    spawn = self._request(
                                        "POST",
                                        f"{self.base_url}/vm/spawn",
                                        json={"machine_id": machine_id},
                                    )
                                    if spawn.status_code not in (200, 201):
                                        self.logger.error("Respawn failed: %s", spawn.text)
                                        return None
                                    current_timeout = int(current_timeout * backoff_factor)
                                    attempt += 1
                                    start = time.time()
                                    self.logger.info(
                                        "Retrying with increased timeout of %ss", current_timeout
                                    )
                                    continue
                                self.logger.error(
                                    "Maximum attempts reached. Machine did not become ready."
                                )
                                return None
                            self.logger.debug("Spawning... (%.0fs elapsed)", elapsed)
                            time.sleep(5)
                            continue
                        self.logger.info("Machine active with IP: %s", active.get("ip"))
                        return active
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Error while waiting: %s", exc)
                time.sleep(4)
                continue

            time.sleep(4)

        self.logger.error("Failed to get machine running after all attempts")
        return None

    # --------------------------- Search/list helpers ---------------------------

    def _search_starting_point(self, machine_name: str) -> dict[str, Any] | None:
        machines: list[dict[str, Any]] = []
        for tier in (1, 2, 3):
            resp = self._request("GET", f"{self.base_url}/sp/tier/{tier}")
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse Starting Point tier %s response", tier)
                continue

            if "data" in data and "machines" in data["data"]:
                for m in data["data"]["machines"]:
                    info = {
                        "id": m["id"],
                        "name": m["name"],
                        "type": "Starting Point",
                        "free": False,
                        "difficulty": "Very Easy",
                    }
                    machines.append(info)
                    if m["name"].lower() == machine_name.lower():
                        self._save_cache("starting_point", {"machines": machines})
                        return info

        if machines:
            self._save_cache("starting_point", {"machines": machines})
        return None

    def _search_active_machines(
        self, machine_name: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        page = 1
        machines: list[dict[str, Any]] = []

        while True:
            data = self._paginated_request(f"{self.base_url}/machine/paginated", page)
            if not data:
                break
            current = data.get("data", [])
            if not current:
                break

            for m in current:
                info = {
                    "id": m["id"],
                    "name": m["name"],
                    "type": "active",
                    "free": m.get("free", False),
                    "difficulty": m.get("difficultyText", "Unknown"),
                    "os": m.get("os", "Unknown"),
                    "points": m.get("points", 0),
                    "rating": m.get("star", 0),
                    "user_owns": m.get("user_owns_count", 0),
                    "root_owns": m.get("root_owns_count", 0),
                    "release": m.get("release"),
                }
                machines.append(info)
                if machine_name and m["name"].lower() == machine_name.lower():
                    self._save_cache("active", {"machines": machines})
                    return info

            page += 1

        if machines:
            self._save_cache("active", {"machines": machines})

        return None if machine_name else machines

    def _search_retired_machines(
        self, machine_name: str | None = None, difficulty: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        page = 1
        machines: list[dict[str, Any]] = []

        if machine_name:
            cached = self._load_cache("retired")
            if cached:
                for m in cached.get("machines", []):
                    if m["name"].lower() == machine_name.lower():
                        self.logger.debug("Found %s in retired cache", machine_name)
                        return m

        while True:
            data = self._paginated_request(f"{self.base_url}/machine/list/retired/paginated", page)
            if not data:
                break
            current = data.get("data", [])
            if not current:
                break

            for m in current:
                if difficulty and m.get("difficultyText", "").lower() == difficulty.lower():
                    pass  # included
                elif difficulty:
                    continue

                info = {
                    "id": m["id"],
                    "name": m["name"],
                    "type": "retired",
                    "free": m.get("free", False),
                    "difficulty": m.get("difficultyText", "Unknown"),
                    "os": m.get("os", "Unknown"),
                    "points": m.get("points", 0),
                    "rating": m.get("star", 0),
                    "user_owns": m.get("user_owns_count", 0),
                    "root_owns": m.get("root_owns_count", 0),
                    "release": m.get("release"),
                }

                if machine_name and m["name"].lower() == machine_name.lower():
                    machines.append(info)
                    self._save_cache("retired", {"machines": machines})
                    return info

                if not machine_name:
                    machines.append(info)

            page += 1

        if machines:
            self._save_cache("retired", {"machines": machines})

        return None if machine_name else machines

    def _find_machine_in_cache(self, machine_name: str) -> dict[str, Any] | None:
        if not self.use_cache:
            return None
        name = machine_name.lower()

        for category in ("starting_point", "active", "retired"):
            cached = self._load_cache(category)
            if not cached:
                continue
            for m in cached.get("machines", []):
                if m["name"].lower() == name:
                    out = dict(m)
                    if category == "retired" and "type" not in out:
                        out["type"] = "retired"
                    return out
        return None

    def _find_machine_in_api(
        self, machine_name: str, machine_type: str | None = None
    ) -> dict[str, Any] | None:
        if machine_type and machine_type.lower() == "retired":
            hit = self._search_retired_machines(machine_name)
            if hit:
                return hit

        hit = self._search_active_machines(machine_name)
        if hit:
            return hit

        hit = self._search_starting_point(machine_name)
        if hit:
            return hit

        if not machine_type or machine_type.lower() != "retired":
            return self._search_retired_machines(machine_name)

        return None

    def get_machine_info(self, machine_name: str) -> dict[str, Any] | None:
        """
        Resolve a machine by name with minimal API calls:
        1) Check currently active machine (fresh)
        2) Check caches
        3) Probe SP -> active -> retired (updates caches)
        """
        active = self._check_active_machine()
        if active and active["name"].lower() == machine_name.lower():
            self.logger.debug("Found %s as active machine", machine_name)
            return active

        cached = self._find_machine_in_cache(machine_name)
        if cached:
            self.logger.debug("Found %s in cache", machine_name)
            return cached

        sp = self._search_starting_point(machine_name)
        if sp:
            self.logger.info("Found %s in Starting Point", machine_name)
            return sp

        act = self._search_active_machines(machine_name)
        if act:
            self.logger.info("Found %s in active machines", machine_name)
            return act

        ret = self._search_retired_machines(machine_name)
        if ret:
            self.logger.info("Found %s in retired machines", machine_name)
            return ret

        return None

    # --------------------------- Listing ---------------------------

    def list_active_machines(self) -> list[dict[str, Any]]:
        machines = self._search_active_machines()  # type: ignore[assignment]
        return machines or []

    def list_retired_machines(self, difficulty: str | None = None) -> list[dict[str, Any]]:
        machines = self._search_retired_machines(None, difficulty)  # type: ignore[arg-type]
        return machines or []

    def list_starting_point_machines(self, tier: int | None = None) -> list[dict[str, Any]]:
        machines: list[dict[str, Any]] = []
        tiers = [tier] if tier is not None else [1, 2, 3]

        for t in tiers:
            resp = self._request("GET", f"{self.base_url}/sp/tier/{t}")
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse Starting Point tier %s response", t)
                continue

            if "data" in data and "machines" in data["data"]:
                for m in data["data"]["machines"]:
                    machines.append(
                        {
                            "id": m["id"],
                            "name": m["name"],
                            "type": "Starting Point",
                            "free": False,
                            "difficulty": "Very Easy",
                        }
                    )

        if machines and self.use_cache:
            self._save_cache("starting_point", {"machines": machines})
        return machines

    # --------------------------- Artifacts & flags ---------------------------

    def download_writeup(self, machine_id: int) -> bytes:
        """Download machine writeup PDF bytes or raise ValueError on failure."""
        resp = self._request("GET", f"{self.base_url}/machine/writeup/{machine_id}")
        if resp.status_code == 200:
            ctype = resp.headers.get("content-type", "")
            if "application/pdf" in ctype:
                return resp.content
            raise ValueError(f"Unexpected content type: {ctype}")
        if resp.status_code == 404:
            raise ValueError("Writeup not available for this machine")
        raise ValueError(f"Failed to download writeup: {resp.text}")

    def get_machine_writeup(self, machine_name: str) -> bytes:
        info = self.get_machine_info(machine_name)
        if not info:
            raise ValueError(f"Machine {machine_name} not found")
        return self.download_writeup(int(info["id"]))

    def submit_flag(self, machine_id: int, flag: str, *, difficulty: int = 50) -> dict[str, Any]:
        """Submit a flag and return normalized result dict."""
        url = f"{self.base_url}/machine/own"
        payload = {"flag": flag, "id": machine_id, "difficulty": difficulty}
        resp = self._request("POST", url, json=payload)

        if resp.status_code == 200 or (
            resp.status_code == 400 and "already owned" in resp.text.lower()
        ):
            return {"success": True, "message": "Flag accepted"}

        try:
            data = resp.json()
            msg = str(data.get("message", "Unknown error"))
            m_low = msg.lower()
            if "already owned" in m_low:
                return {"success": True, "message": msg}
            if "incorrect flag" in m_low:
                return {"success": False, "message": msg}
            return {"success": False, "message": msg}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Error parsing response: %s", exc)
            return {"success": False, "message": f"Error parsing response: {exc!s}"}

    def get_vpn_config(self, server_id: int = 1) -> Path:
        """
        Ensure VPN configuration file exists locally:
        1) Switch server
        2) Download ovpn file
        3) Save under ./vpn/vpn_config_{server_id}.ovpn
        """
        vpn_file = self.vpn_dir / f"vpn_config_{server_id}.ovpn"
        if vpn_file.exists():
            self.logger.debug("Using cached VPN config for server %s", server_id)
            return vpn_file

        self.logger.info("Downloading VPN config for server %s", server_id)

        switch = self._request("POST", f"{self.base_url}/connections/servers/switch/{server_id}")
        if switch.status_code not in (200, 201):
            raise RuntimeError(f"Failed to switch VPN server: {switch.text}")

        cfg = self._request("GET", f"{self.base_url}/access/ovpnfile/{server_id}/0")
        if cfg.status_code != 200:
            raise RuntimeError(f"Failed to download VPN config: {cfg.text}")

        vpn_file.write_bytes(cfg.content)
        self.logger.info("VPN config saved to %s", vpn_file)
        return vpn_file


# --------------------------- CLI ---------------------------


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup console logging for the client CLI."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the HTB client."""
    parser = argparse.ArgumentParser(description="HackTheBox CLI")
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("spawn", help="Spawn a machine")
    sp.add_argument("machine_id", type=int, help="ID of the machine to spawn")
    sp.add_argument("machine_name", help="Name of the machine to spawn")

    info = sub.add_parser("info", help="Get detailed information about a machine")
    info.add_argument("machine_name", help="Name of the machine")

    ls = sub.add_parser("list", help="List machines")
    ls.add_argument(
        "--category",
        choices=["active", "retired", "starting_point"],
        default="active",
        help="Which list to show",
    )
    ls.add_argument("--difficulty", help="Filter retired by difficulty")
    ls.add_argument("--tier", type=int, choices=[1, 2, 3], help="Starting Point tier")
    ls.add_argument("--json", action="store_true", help="Minified JSON output")

    parser.add_argument("--debug", action="store_true", help="Verbose debug logs")
    parser.add_argument("--token", help="HTB API token (or set HTB_TOKEN)")
    parser.add_argument("--nocache", action="store_true", help="Disable cache usage")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    logger = setup_logging(args.debug)

    token = args.token or os.getenv("HTB_TOKEN")
    if not token:
        logger.error("No HTB token provided. Use --token or set HTB_TOKEN environment variable.")
        sys.exit(1)

    client = HTBClient(token=token, use_cache=not args.nocache, debug=args.debug)

    try:
        if args.command == "spawn":
            info = client.spawn_machine(args.machine_id, args.machine_name)
            if info and info.get("ip"):
                print(info["ip"])
            else:
                logger.error("No IP address found")
                sys.exit(1)

        elif args.command == "info":
            info = client.get_machine_info(args.machine_name)
            if info:
                print(json.dumps(info, indent=None if args.json else 2))
            else:
                logger.error("Machine %s not found", args.machine_name)
                sys.exit(1)

        elif args.command == "list":
            result: dict[str, Any] = {"category": args.category, "machines": []}
            if args.category == "active":
                result["machines"] = client.list_active_machines()
            elif args.category == "retired":
                result["machines"] = client.list_retired_machines(args.difficulty)
                if args.difficulty:
                    result["difficulty_filter"] = args.difficulty
            else:
                result["machines"] = client.list_starting_point_machines(args.tier)
                if args.tier:
                    result["tier_filter"] = args.tier

            print(json.dumps(result, indent=None if args.json else 2))

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        logger.error("Error: %s", exc)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
