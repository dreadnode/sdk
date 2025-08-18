import json
import os
import shlex
import shutil
import subprocess
import typing as t
from pathlib import Path

from pydantic import Field

from dreadnode.agent.tools import Toolset, tool_method


def _which(exe: str) -> str | None:
    return shutil.which(exe) or (exe if Path(exe).exists() else None)


def _parse_jsonl(text: str) -> list[dict]:
    items: list[dict] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            items.append(json.loads(ln))
        except json.JSONDecodeError:
            # tolerate occasional non-JSON lines when -silent wasn't honored
            continue
    return items


class ProjectDiscoveryTools(Toolset):
    """
    Wrappers for ProjectDiscovery CLIs:
      nuclei, katana, httpx (PD CLI), subfinder, cvemap, uncover, naabu.

    All methods return structured dictionaries with parsed JSON(L) results and the exact command run.
    """

    # ---- binary names/paths (override if installed elsewhere)
    nuclei_cmd: str = Field(default="nuclei")
    katana_cmd: str = Field(default="katana")
    httpx_cmd: str = Field(default="httpx")  # PD CLI (not the Python library)
    subfinder_cmd: str = Field(default="subfinder")
    cvemap_cmd: str = Field(default="cvemap")
    uncover_cmd: str = Field(default="uncover")
    naabu_cmd: str = Field(default="naabu")

    # ---- common toggles
    prefer_silent: bool = Field(
        default=True, description="Add -silent where supported to keep stdout pure JSON."
    )
    max_output: int | None = Field(
        default=None, description="Truncate parsed items to this many (None = no limit)."
    )

    # ---- helpers

    def _run(
        self, cmd: list[str], *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(cmd, text=True, capture_output=True, env=full_env, check=False)

    def _ensure(self, exe: str, name: str):
        if not _which(exe):
            raise RuntimeError(f"{name} not found on PATH (looked for '{exe}')")

    def _limit(self, items: list[dict]) -> list[dict]:
        return items[: self.max_output] if (self.max_output and self.max_output > 0) else items

    # ---------------- nuclei ----------------

    @tool_method(
        name="pd_nuclei_scan",
        description="Run nuclei with JSONL output; supports -u or -l and templates/tags/severity.",
        variants=["all", "security", "recon"],
        catch=(Exception,),
    )
    def nuclei_scan(
        self,
        *,
        url: str | None = None,
        list_file: str | None = None,
        templates: list[str] | None = None,  # e.g., ["cves/", "exposures/"]
        template_ids: list[str] | None = None,  # e.g., ["CVE-2022-XXXXX"]
        tags: list[str] | None = None,  # e.g., ["cve,exposure"]
        severity: str | None = None,  # e.g., "critical,high"
        rate_limit: int | None = None,  # -rl
        concurrency: int | None = None,  # -c
        headless: bool = False,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.nuclei_cmd, "nuclei")
        if not url and not list_file:
            raise ValueError("Provide url= or list_file= for nuclei input.")

        cmd = [self.nuclei_cmd]
        if url:
            cmd += ["-u", url]
        if list_file:
            cmd += ["-l", list_file]
        if templates:
            for tpath in templates:
                cmd += ["-t", tpath]
        if template_ids:
            for tid in template_ids:
                cmd += ["-id", tid]
        if tags:
            cmd += ["-tags", ",".join(tags)]
        if severity:
            cmd += ["-severity", severity]
        if rate_limit:
            cmd += ["-rl", str(rate_limit)]
        if concurrency:
            cmd += ["-c", str(concurrency)]
        if headless:
            cmd += ["-headless"]
        if self.prefer_silent:
            cmd += ["-silent"]

        # JSONL to stdout
        cmd += ["-j"]

        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"nuclei failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- katana ----------------

    @tool_method(
        name="pd_katana_crawl",
        description="Run katana with -jsonl; returns parsed URL records.",
        variants=["all", "recon"],
        catch=(Exception,),
    )
    def katana_crawl(
        self,
        target: str,  # URL or a file (auto-detected)
        *,
        depth: int = 3,  # -d
        scope: t.Literal["rdn", "fqdn", "dn"] = "rdn",  # -fs
        js_crawl: bool = True,  # -jc
        rate_limit: int | None = None,  # -rl
        concurrency: int | None = None,  # -c
        headers_file: str | None = None,  # -H (file)
        extra_args: list[str] | None = None,
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.katana_cmd, "katana")

        cmd = [self.katana_cmd, "-jsonl", "-d", str(depth), "-fs", scope]
        if target.startswith(("http://", "https://")):
            cmd += ["-u", target]
        else:
            cmd += ["-list", target]
        if js_crawl:
            cmd += ["-jc"]
        if rate_limit:
            cmd += ["-rl", str(rate_limit)]
        if concurrency:
            cmd += ["-c", str(concurrency)]
        if headers_file:
            cmd += ["-H", headers_file]
        if self.prefer_silent:
            cmd += ["-silent"]
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"katana failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- httpx (PD CLI) ----------------

    @tool_method(
        name="pd_httpx_probe",
        description="Run PD httpx with -json; returns JSONL results (status/title/etc depending on flags).",
        variants=["all", "recon"],
        catch=(Exception,),
    )
    def httpx_probe(
        self,
        *,
        url: str | None = None,
        list_file: str | None = None,
        probes: list[str] | None = None,  # e.g., ["-status-code","-title","-tech-detect"]
        rate_limit: int | None = None,  # -rl
        concurrency: int | None = None,  # -c
        extra_args: list[str] | None = None,
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.httpx_cmd, "httpx (ProjectDiscovery CLI)")
        if not url and not list_file:
            raise ValueError("Provide url= or list_file= for httpx input.")

        cmd = [self.httpx_cmd, "-json"]
        if self.prefer_silent:
            cmd += ["-silent"]
        if url:
            cmd += ["-u", url]
        if list_file:
            cmd += ["-l", list_file]
        if rate_limit:
            cmd += ["-rl", str(rate_limit)]
        if concurrency:
            cmd += ["-c", str(concurrency)]
        if probes:
            cmd += probes
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"httpx failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- subfinder ----------------

    @tool_method(
        name="pd_subfinder",
        description="Run subfinder with -json; returns discovered subdomains (JSONL).",
        variants=["all", "recon"],
        catch=(Exception,),
    )
    def subfinder(
        self,
        *,
        domain: str | None = None,
        domains_file: str | None = None,
        all_sources: bool = False,  # -all
        recursive: bool = False,  # -recursive
        rate_limit: int | None = None,  # -rl
        extra_args: list[str] | None = None,
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.subfinder_cmd, "subfinder")
        if not domain and not domains_file:
            raise ValueError("Provide domain= or domains_file= for subfinder input.")

        cmd = [self.subfinder_cmd, "-json"]
        if self.prefer_silent:
            cmd += ["-silent"]
        if domain:
            cmd += ["-d", domain]
        if domains_file:
            cmd += ["-dL", domains_file]
        if all_sources:
            cmd += ["-all"]
        if recursive:
            cmd += ["-recursive"]
        if rate_limit:
            cmd += ["-rl", str(rate_limit)]
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"subfinder failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- naabu ----------------

    @tool_method(
        name="pd_naabu_scan",
        description="Run naabu with -json (JSONL); returns open ports.",
        variants=["all", "recon"],
        catch=(Exception,),
    )
    def naabu_scan(
        self,
        *,
        host: str | None = None,  # single host/IP
        hosts_file: str | None = None,  # -list
        ports: str | None = None,  # -p "80,443,8080" or "top-1000"
        rate: int | None = None,  # -rate
        exclude_cdn: bool = False,  # -exclude-cdn
        extra_args: list[str] | None = None,
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.naabu_cmd, "naabu")
        if not host and not hosts_file:
            raise ValueError("Provide host= or hosts_file= for naabu input.")

        cmd = [self.naabu_cmd, "-json"]
        if self.prefer_silent:
            cmd += ["-silent"]
        if host:
            cmd += ["-host", host]
        if hosts_file:
            cmd += ["-list", hosts_file]
        if ports:
            cmd += ["-p", ports]
        if rate:
            cmd += ["-rate", str(rate)]
        if exclude_cdn:
            cmd += ["-exclude-cdn"]
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"naabu failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- uncover ----------------

    @tool_method(
        name="pd_uncover",
        description="Run uncover (passive search engines) with -json; returns JSONL results.",
        variants=["all", "recon"],
        catch=(Exception,),
    )
    def uncover(
        self,
        query: str,
        *,
        engines: list[str]
        | None = None,  # -e shodan,censys,fofa,zoomeye,quake,netlas,criminalip,hunter,publicwww ...
        limit: int | None = None,  # -l
        field: str | None = None,  # -f (ip,port,host or custom template)
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,  # supply provider API keys via env if needed
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.uncover_cmd, "uncover")

        cmd = [self.uncover_cmd, "-j", "-q", query]
        if self.prefer_silent:
            cmd += ["-silent"]
        if engines:
            cmd += ["-e", ",".join(engines)]
        if limit:
            cmd += ["-l", str(limit)]
        if field:
            cmd += ["-f", field]
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"uncover failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout)
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}

    # ---------------- cvemap ----------------

    @tool_method(
        name="pd_cvemap",
        description="Run cvemap with -json; query CVE data (requires PDCP API key).",
        variants=["all", "recon", "security"],
        catch=(Exception,),
    )
    def cvemap(
        self,
        *,
        ids: list[str] | None = None,  # CVE-YYYY-NNNN ...
        search: str
        | None = None,  # free-text / fielded query, e.g., 'severity:critical vendor:oracle'
        limit: int | None = None,  # -l
        offset: int | None = None,  # -o
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,  # include {"PDCP_API_KEY": "..."} or per-docs
    ) -> dict:
        """
        Returns: {"items":[...], "cmd":[...], "stderr":"..."}
        """
        self._ensure(self.cvemap_cmd, "cvemap")

        cmd = [self.cvemap_cmd, "-json"]
        if ids:
            for cid in ids:
                cmd += ["-id", cid]
        if search:
            cmd += ["-q", search]
        if limit is not None:
            cmd += ["-l", str(limit)]
        if offset is not None:
            cmd += ["-o", str(offset)]
        if extra_args:
            cmd += list(extra_args)

        proc = self._run(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cvemap failed rc={proc.returncode}\ncmd={shlex.join(cmd)}\n{proc.stderr}"
            )

        items = _parse_jsonl(proc.stdout) or (
            json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else []
        )
        return {"items": self._limit(items), "cmd": cmd, "stderr": proc.stderr}
