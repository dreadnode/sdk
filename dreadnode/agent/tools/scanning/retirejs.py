import json
import os
import shlex
import shutil
import subprocess
import tempfile
import typing as t
from pathlib import Path

from pydantic import Field

from dreadnode.agent.tools import Toolset, tool_method


class RetireJS(Toolset):
    """
    A thin, robust wrapper around retire.js.

    - Runs via `npx` with a pinned version (default) for reproducibility.
    - Writes machine-readable output to a temp file, then returns parsed JSON.
    - Normalizes exit behavior so vulnerabilities don't crash the caller by default.
    """

    npx: str = Field(default="npx", description="Path or command name for npx.")
    version: str = Field(default="5.3.0", description="retire.js version to pin (e.g., 5.3.0).")
    exitwith: int = Field(
        default=0,
        description="Override retire.js exit code for findings (use 13 to enforce failure).",
    )
    default_mode: t.Literal["both", "js", "node"] = Field(
        default="both", description="Scan mode to use by default."
    )
    default_severity: t.Literal["none", "low", "medium", "high", "critical"] = Field(
        default="none", description="Severity threshold; controls retire's finding gate."
    )
    allow_stdout_fallback: bool = Field(
        default=True,
        description="If the output file is missing, return stdout/stderr payload instead of hard-failing.",
    )

    def _which(self, exe: str) -> str:
        return shutil.which(exe) or exe

    def _base_cmd(self) -> list[str]:
        return [self._which(self.npx), "--yes", f"retire@{self.version}"]

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        # Inherit HTTP(S)_PROXY etc if present
        env = os.environ.copy()
        return subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)

    def _read_json_if_exists(self, path: Path) -> dict[str, t.Any] | None:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse retire.js JSON at {path}: {e}") from e
        return None

    @tool_method(
        name="retire_scan",
        description="Scan a directory for vulnerable JS and Node deps using retire.js; returns parsed JSON (or stdout fallback).",
        catch=(Exception,),
    )
    def scan(
        self,
        path: str | Path,
        mode: t.Literal["both", "js", "node"] | None = None,
        severity: t.Literal["none", "low", "medium", "high", "critical"] | None = None,
        ignorefile: str | Path | None = None,
        extra_args: list[str] | None = None,
        outputformat: t.Literal["json", "cyclonedx"] = "json",
    ) -> dict:
        """
        Run retire.js against `path`.

        Args:
            path: Directory to scan.
            mode: "both" (default), "js" (raw/vendored JS), or "node" (Node deps).
            severity: Finding threshold. Default is the tool's configured `default_severity`.
            ignorefile: Path to `.retireignore` / `.retireignore.json`.
            extra_args: Additional raw CLI flags (advanced use).
            outputformat: "json" (default) or "cyclonedx" SBOM.

        Returns:
            A dict with keys:
            - "cmd": The exact command run (for auditing).
            - "report": Parsed JSON if produced.
            - "stdout"/"stderr": Captured text (useful on fallback or debugging).
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        run_mode = mode or self.default_mode
        sev = severity or self.default_severity

        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / (
                "retire-report.cdx.xml" if outputformat == "cyclonedx" else "retire-report.json"
            )

            cmd: list[str] = self._base_cmd()

            # Scope flags
            if run_mode == "js":
                cmd += ["--js", "--path", str(p)]
            elif run_mode == "node":
                cmd += ["--node", "--path", str(p)]
            else:  # both (default retire behavior scans both when only --path is given)
                cmd += ["--path", str(p)]

            # Output
            cmd += ["--outputformat", outputformat, "--outputpath", str(out_file)]
            # Normalize exit behavior (13 means "findings", not an actual failure)
            cmd += ["--exitwith", str(self.exitwith)]
            # Controls gate by severity (retire honors this for exit code logic)
            if sev:
                cmd += ["--severity", sev]
            # Optional ignorefile
            if ignorefile:
                cmd += ["--ignorefile", str(Path(ignorefile).resolve())]
            # Extra pass-through flags (e.g., ["--nocache"])
            if extra_args:
                cmd += list(extra_args)

            proc = self._run(cmd)

            # If caller chose to enforce failures (e.g., exitwith=13), surface non-zero RC except 13 when exitwith!=0.
            if proc.returncode != 0 and self.exitwith == 0:
                # When we force exitwith=0, any non-zero here is a genuine execution error.
                raise RuntimeError(
                    "retire.js invocation failed.\n"
                    f"rc={proc.returncode}\ncmd={shlex.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )

            report = self._read_json_if_exists(out_file) if outputformat == "json" else None

            if report is None and self.allow_stdout_fallback:
                return {
                    "cmd": cmd,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "outputformat": outputformat,
                }

            return {
                "cmd": cmd,
                "report": report,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "outputformat": outputformat,
            }

    @tool_method(
        name="retire_sbom",
        description="Generate a CycloneDX SBOM with retire.js and return the SBOM file path.",
        catch=(Exception,),
        truncate=None,
        variants=["all", "security"],
    )
    def sbom(
        self,
        path: str | Path,
        out_file: str | Path = "retire-sbom.cdx.xml",
        mode: t.Literal["both", "js", "node"] | None = None,
        ignorefile: str | Path | None = None,
        extra_args: list[str] | None = None,
    ) -> dict:
        """
        Produce a CycloneDX SBOM using retire.js and write it to `out_file`.

        Args:
            path: Directory to scan.
            out_file: Output filename (written in CWD, unless absolute).
            mode: "both" (default), "js", or "node".
            ignorefile: Path to `.retireignore` / `.retireignore.json`.
            extra_args: Additional raw CLI flags.

        Returns:
            {"cmd": [...], "sbom_path": "<path>", "stdout": "...", "stderr": "..."}
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        run_mode = mode or self.default_mode
        out_path = Path(out_file).resolve()

        # Write to a temp file first, then move atomically into place
        with tempfile.TemporaryDirectory() as td:
            tmp_out = Path(td) / "retire-sbom.cdx.xml"

            cmd: list[str] = self._base_cmd()
            if run_mode == "js":
                cmd += ["--js", "--path", str(p)]
            elif run_mode == "node":
                cmd += ["--node", "--path", str(p)]
            else:
                cmd += ["--path", str(p)]

            cmd += [
                "--outputformat",
                "cyclonedx",
                "--outputpath",
                str(tmp_out),
                "--exitwith",
                str(self.exitwith),
            ]
            if ignorefile:
                cmd += ["--ignorefile", str(Path(ignorefile).resolve())]
            if extra_args:
                cmd += list(extra_args)

            proc = self._run(cmd)
            if proc.returncode != 0 and self.exitwith == 0:
                raise RuntimeError(
                    "retire.js SBOM generation failed.\n"
                    f"rc={proc.returncode}\ncmd={shlex.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )

            # Move into place
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if tmp_out.exists():
                out_path.write_bytes(tmp_out.read_bytes())

        return {
            "cmd": cmd,
            "sbom_path": str(out_path),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    @tool_method(
        name="retire_version",
        description="Return retire.js CLI version (from the pinned npx spec).",
        catch=(Exception,),
        truncate=None,
        variants=["all", "security"],
    )
    def version_info(self) -> dict:
        """
        Returns retire.js version information using `retire --version`.
        """
        cmd = self._base_cmd() + ["--version"]
        proc = self._run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to get retire.js version.\nrc={proc.returncode}\ncmd={shlex.join(cmd)}\nstderr={proc.stderr}"
            )
        return {"cmd": cmd, "stdout": proc.stdout.strip()}
