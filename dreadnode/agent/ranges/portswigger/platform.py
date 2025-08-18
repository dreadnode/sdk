import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import httpx
from loguru import logger
from selectolax.parser import HTMLParser

from dreadnode.agent.tools import Toolset

BASE_URL: Final[str] = "https://portswigger.net"
LABS_URL: Final[str] = f"{BASE_URL}/web-security/all-labs"
DEFAULT_UA: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(slots=True, kw_only=True)
class PortSwiggerLab:
    """Represents a single PortSwigger lab."""

    title: str
    url: str
    difficulty: str
    category: str | None = None
    lab_id: str | None = None
    instance_url: str | None = None
    description: str | None = None

    def __str__(self) -> str:
        return f"{self.title} ({self.difficulty})"

    @property
    def path(self) -> str | None:
        """Path component of the lab URL (relative to BASE_URL)."""
        if not self.url:
            return None
        return self.url.replace(BASE_URL, "").lstrip("/")

    def get_launch_url(self) -> str | None:
        """URL to launch the lab instance."""
        if not self.lab_id:
            return None
        return f"{BASE_URL}/academy/labs/launch/{self.lab_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortSwiggerLab":
        return cls(**data)


class PortSwiggerPlatform:
    """Interact with PortSwigger Web Security Academy labs."""

    def __init__(
        self,
        attempts_dir: str | Path,
        executor: Toolset | None = None,
        *,
        use_cache: bool = True,
        keep_target: bool = False,
        timeout_s: float = 20.0,
    ) -> None:
        """
        Args:
            attempts_dir: Directory to store attempt data and cache
            executor: Optional executor used by BasePlatform
            use_cache: Whether to use on-disk cache for labs list
            keep_target: If True, skip cleanup (labs auto-terminate anyway)
            timeout_s: HTTP timeout for requests
        """
        super().__init__(str(attempts_dir), executor)

        self._client = httpx.Client(
            headers={"User-Agent": DEFAULT_UA},
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=False,  # we inspect redirects manually
            http2=True,
        )
        self._authenticated: bool = False
        self.keep_target: bool = keep_target
        self.use_cache: bool = use_cache

        self.attempts_dir = Path(attempts_dir)
        self.cache_dir = self.attempts_dir / ".apicache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._labs_cache_file = self.cache_dir / "labs.json"
        self.labs: list[PortSwiggerLab] = []
        if self.use_cache:
            self._load_labs_from_cache()

    def _load_labs_from_cache(self) -> None:
        if not self._labs_cache_file.exists():
            return
        try:
            raw = self._labs_cache_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.labs = [PortSwiggerLab.from_dict(d) for d in data]
            logger.info("Loaded %d labs from cache", len(self.labs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load labs cache: %s", exc, exc_info=False)
            self.labs = []

    def _save_labs_to_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            payload = [lab.to_dict() for lab in self.labs]
            self._labs_cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("Saved %d labs to cache", len(self.labs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save labs cache: %s", exc, exc_info=False)

    def _find_lab_in_cache(self, title: str) -> PortSwiggerLab | None:
        if not self.labs and self.use_cache:
            self._load_labs_from_cache()
        needle = title.lower()
        return next((lab for lab in self.labs if needle in lab.title.lower()), None)

    # -------------------------- Auth --------------------------

    def _authenticate(self) -> bool:
        """Authenticate using SecretManager credentials and set cookies."""
        if self._authenticated:
            return True

        username, password = self.secrets_manager.get_portswigger_username_and_password()
        logger.info("Authenticating with PortSwigger as %s", username)

        login_url = f"{BASE_URL}/users"

        # Fetch login page to grab CSRF token
        resp = self._client.get(login_url)
        if resp.status_code != 200:
            logger.error("Login page returned %s", resp.status_code)
            return False

        token = self._extract_csrf_token(resp.text)
        if not token:
            logger.error("CSRF token not found on login page")
            return False

        form_data = {
            "RequestVerificationToken": token,
            "EmailAddress": username,
            "Password": password,
            "RememberMe": "false",
            "ajaxRequest": "true",
        }
        post = self._client.post(login_url, data=form_data)

        if post.status_code not in (200, 302, 303):
            logger.error("Login POST failed with %s", post.status_code)
            return False

        # httpx manages cookies automatically; verify presence of expected cookies
        jar = self._client.cookies
        if not (jar.get("SessionId", domain="portswigger.net") or jar.get("SessionId")):
            logger.error("SessionId cookie missing after login")
            return False

        # Some flows rely on this cookie; if it's set, great
        if jar.get("Authenticated_UserVerificationId", domain="portswigger.net"):
            logger.debug("Authenticated_UserVerificationId cookie present")

        self._authenticated = True
        logger.info("Authenticated successfully")
        return True

    @staticmethod
    def _extract_csrf_token(html: str) -> str | None:
        """Extract RequestVerificationToken from login page."""
        root = HTMLParser(html)
        node = root.css_first('input[name="RequestVerificationToken"]')
        return node.attributes.get("value") if node else None

    # -------------------------- Fetch & Parse --------------------------

    def fetch_labs(self) -> list[PortSwiggerLab]:
        """Fetch available labs from PortSwigger."""
        if self.use_cache and self.labs:
            return self.labs

        try:
            logger.info("Fetching labs from %s", LABS_URL)
            resp = self._client.get(LABS_URL)
            resp.raise_for_status()

            root = HTMLParser(resp.text)
            containers = root.css("div.widgetcontainer-lab-link")
            labs: list[PortSwiggerLab] = []

            for c in containers:
                difficulty_node = next(
                    (
                        n
                        for n in c.css("span")
                        if n.attributes.get("class", "").startswith("label-")
                    ),
                    None,
                )
                difficulty = difficulty_node.text(strip=True) if difficulty_node else "Unknown"

                link = c.css_first("a")
                if not link:
                    continue

                title = link.text(strip=True)
                href = link.attributes.get("href", "")
                url = href if href.startswith("http") else f"{BASE_URL}{href}"

                # Category derived from URL segment after /web-security/
                category = "Web Security"
                if "/web-security/" in url:
                    after = url.split("/web-security/", 1)[1]
                    category = after.split("/", 1)[0].replace("-", " ").title()

                labs.append(
                    PortSwiggerLab(title=title, url=url, difficulty=difficulty, category=category)
                )

            self.labs = labs
            self._save_labs_to_cache()
            logger.info("Fetched %d labs", len(labs))
            return labs

        except httpx.HTTPError as exc:
            logger.error("Error fetching labs: %s", exc)
            return []

    # -------------------------- Lookup --------------------------

    def list_labs(self) -> None:
        """Print labs as JSON to stdout (for CLI use)."""
        if not self.labs:
            self.fetch_labs()
        payload = [
            {
                "name": lab.title,
                "difficulty": lab.difficulty,
                "category": lab.category or "Web Security",
                "url": lab.url,
                "instance_url": lab.instance_url,
            }
            for lab in self.labs
        ]
        print(json.dumps(payload, indent=2))

    def find_lab_by_title(self, title: str) -> PortSwiggerLab | None:
        """Case-insensitive partial title match."""
        lab = self._find_lab_in_cache(title)
        if lab:
            logger.debug("Found lab in cache: %s", lab.title)
            return lab

        if not self.labs:
            self.fetch_labs()

        needle = title.lower()
        matches = [l for l in self.labs if needle in l.title.lower()]
        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "Multiple labs match title '%s': %s", title, ", ".join(l.title for l in matches)
            )
        return matches[0]

    def find_lab_by_url(self, url: str) -> PortSwiggerLab | None:
        if not self.labs:
            self.fetch_labs()
        full = url if url.startswith(BASE_URL) else f"{BASE_URL}{url}"
        return next((l for l in self.labs if l.url == full), None)

    @staticmethod
    def _extract_lab_description(root: HTMLParser, *, fallback_title: str) -> str:
        """
        Extract the description that appears between the share icon and the
        left buttons container. If structure shifts, fall back gracefully.
        """
        share = root.css_first("span.icon-share")
        buttons = root.css_first("div.container-buttons-left")

        # Best-effort "between" extraction
        if share and buttons:
            # Traverse entire document in order and capture nodes between markers.
            nodes: list[Any] = list(root.root.traverse()) if root.root else []
            try:
                i_share = nodes.index(share.node)
                i_buttons = nodes.index(buttons.node)
                between = nodes[i_share + 1 : i_buttons]
                pieces: list[str] = []
                for n in between:
                    tag = getattr(n, "tag", None)
                    if tag in {"p", "code"}:
                        # Selectolax Node.as_html() is `n.html`, text is `n.text()`
                        html = getattr(n, "html", None)
                        pieces.append(html if html else n.text(deep=True, separator=" ").strip())
                if pieces:
                    desc = "".join(pieces)
                    logger.info("Extracted description (%d chars)", len(desc))
                    return desc
            except ValueError:
                logger.debug("Markers not found in traversal; falling back")

        # Secondary fallback: grab first descriptive block near header
        intro = root.css_first("div.widget-lab-intro, article, section")
        if intro:
            paras = [p.text(strip=True) for p in intro.css("p") if p.text(strip=True)]
            if paras:
                return "\n\n".join(paras[:4])

        return f"PortSwigger Web Security Academy Lab: {fallback_title}"

    # -------------------------- Access / Launch --------------------------

    def access_lab(self, lab: PortSwiggerLab) -> str:
        """Launch a lab instance and return the instance URL."""
        # Fetch lab page
        resp = self._client.get(lab.url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to GET lab page ({resp.status_code})")

        root = HTMLParser(resp.text)

        # Description
        lab.description = self._extract_lab_description(root, fallback_title=lab.title)

        # Launch button -> lab_id
        launch = root.css_first("a.button-orange")
        if not launch:
            raise RuntimeError("Lab launch button not found")
        href = launch.attributes.get("href", "")
        lab_id = href.strip("/").split("/")[-1]
        lab.lab_id = lab_id

        launch_url = f"{BASE_URL}/academy/labs/launch/{lab_id}"
        ref = lab.path or ""
        launch_with_ref = f"{launch_url}?referrer={ref}"

        logger.info("Launching lab: %s", lab.title)
        for attempt in range(1, 4):
            try:
                # Minimize headers to mimic browser redirect behavior
                headers = {"Host": "portswigger.net"}
                resp = self._client.get(launch_with_ref, headers=headers)

                # Success path: redirect to web-security-academy.net instance
                if resp.is_redirect or resp.is_client_error or resp.is_server_error:
                    loc = resp.headers.get("location", "")
                    if "web-security-academy.net" in loc:
                        lab.instance_url = loc
                        logger.info("Lab instance URL: %s", loc)
                        return loc

                    # Re-auth required
                    if "/users" in loc:
                        logger.info("Re-auth required, attempting authentication")
                        if not self._authenticate():
                            raise RuntimeError("Re-authentication failed")

                        # retry next loop
                        time.sleep(1.0)
                        continue

                # Sometimes the instance is provisioned; try following redirect manually
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location", "")
                    if "web-security-academy.net" in loc:
                        lab.instance_url = loc
                        logger.info("Lab instance URL: %s", loc)
                        return loc

                logger.debug("Attempt %d: status=%s", attempt, resp.status_code)
                time.sleep(2.0)

            except httpx.HTTPError as exc:
                logger.warning("Launch attempt %d failed: %s", attempt, exc)
                time.sleep(2.0)

        raise RuntimeError("Failed to launch lab after 3 attempts")

    # -------------------------- Targets API --------------------------

    def initialize_target(self, target_name: str) -> PortSwiggerLab:
        """
        Initialize a lab and return a ready Target.
        """
        self._authenticate()
        lab = self.find_lab_by_title(target_name)
        if not lab:
            raise ValueError(f"Lab not found: {target_name}")

        instance_url = self.access_lab(lab)
        if not instance_url:
            raise RuntimeError(f"Failed to launch lab: {target_name}")

        target = PortSwiggerLab(
            name=lab.title,
            identifier=lab.url,
            type="web",
            difficulty=lab.difficulty,
            is_active=True,
            is_ready=True,
            connection_info=instance_url,
            metadata={
                "category": lab.category or "Web Security",
                "description": lab.description
                or f"PortSwigger Web Security Academy Lab: {lab.title}",
                "url": lab.url,
            },
        )
        self.target = target
        return target

    def list_targets(self) -> list[dict[str, Any]]:
        if not self.labs:
            self.fetch_labs()
        return [
            {
                "name": lab.title,
                "difficulty": lab.difficulty,
                "category": lab.category or "Web Security",
                "url": lab.url,
                "status": "available",
            }
            for lab in self.labs
        ]

    @property
    def platform_name(self) -> str:
        return "PortSwigger"

    # TODO: Implement a hook for flag validation
    def validate_flag(self, flag: str, target: PortSwiggerLab | None = None) -> bool:  # noqa: ARG002
        """No traditional flags; treat any non-empty string as 'has answer'."""
        return bool(flag and flag.strip())


# -------------------------- CLI --------------------------


def _default_attempts_dir() -> Path:
    return Path.home() / ".cache" / "boxpwnr" / "portswigger"


def main() -> None:
    parser = argparse.ArgumentParser(description="PortSwigger Web Security Academy Labs CLI")
    parser.add_argument("--list", action="store_true", help="List all available labs")
    parser.add_argument("--title", help="Access lab by title (partial match)")
    parser.add_argument("--url", help="Access lab by URL")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--attempts-dir",
        type=Path,
        default=_default_attempts_dir(),
        help=f"Directory to store attempts/cache (default: {_default_attempts_dir()})",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity (-v, -vv)"
    )
    args = parser.parse_args()

    # Basic per-run logging setup (kept in CLI, not in library)
    level = (
        logging.WARNING
        if args.verbose == 0
        else (logging.INFO if args.verbose == 1 else logging.DEBUG)
    )
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    platform = PortSwiggerPlatform(
        attempts_dir=args.attempts_dir,
        use_cache=not args.no_cache,
    )

    if args.list:
        platform.list_labs()
        return

    if args.title:
        lab = platform.find_lab_by_title(args.title)
        if not lab:
            print(f"No lab found matching title: {args.title}")
            return
        try:
            url = platform.access_lab(lab)
            print(f"Lab instance URL: {url}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to access lab instance: {exc}")
        return

    if args.url:
        lab = platform.find_lab_by_url(args.url)
        if not lab:
            print(f"No lab found matching URL: {args.url}")
            return
        try:
            url = platform.access_lab(lab)
            print(f"Lab instance URL: {url}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to access lab instance: {exc}")
        return

    # Default: fetch & group by category
    print("Fetching PortSwigger labs...")
    labs = platform.fetch_labs()
    if not labs:
        print("No labs found or failed to fetch labs.")
        return

    by_cat: dict[str | None, list[PortSwiggerLab]] = {}
    for lab in labs:
        by_cat.setdefault(lab.category, []).append(lab)

    total = sum(len(v) for v in by_cat.values())
    print(f"\nFound {total} labs across {len(by_cat)} categories:\n")
    for category, items in sorted(by_cat.items(), key=lambda kv: (kv[0] or "")):
        print(f"\n{category or 'Web Security'} ({len(items)} labs):")
        for lab in items:
            print(f"  - {lab}")


if __name__ == "__main__":
    main()
