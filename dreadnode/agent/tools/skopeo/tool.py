import json
import subprocess
import tarfile
import typing as t
import zlib
from io import BytesIO
from pathlib import Path

import httpx

from dreadnode.agent.tools.base import Toolset, tool_method


class Skopeo(Toolset):
    """
    Tools for inspecting Microsoft Container Registry images via skopeo + httpx.
    """

    name = "Skopeo"

    registry: str = "mcr.microsoft.com"
    chunk_size: int = 10 * 1024
    max_attempts: int = 10
    default_out_dir: str = "/workspace/out"

    def _run(self, cmd: str) -> subprocess.CompletedProcess:
        cmd_list = cmd.split() if isinstance(cmd, str) else cmd
        return subprocess.run(cmd_list, shell=False, capture_output=True, check=False)  # noqa: S603

    def _skopeo_json(self, args: str) -> dict[str, t.Any]:
        cp = self._run(f"skopeo {args}")
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.decode() or "skopeo failed")
        return json.loads(cp.stdout or "{}")

    def _peek_docker_layer(self, repo: str, digest: str) -> list[str]:
        """
        Progressive, partial Range reads of a gzipped tar layer to list file names without
        downloading the whole blob.
        """
        files: list[str] = []
        bytes_read: int = 0
        chunk_size = self.chunk_size
        buffer = BytesIO()
        url = f"https://{self.registry}/v2/{repo}/blobs/{digest}"

        for _ in range(self.max_attempts):
            range_end = bytes_read + chunk_size - 1
            headers = {"Range": f"bytes={bytes_read}-{range_end}"}
            chunk_size *= 2

            with httpx.get(url, headers=headers, stream=True) as r:
                if r.status_code not in (200, 206):
                    break
                buffer.seek(0, 2)
                buffer.write(r.content)
                bytes_read += len(r.content)
                buffer.seek(0)

            # GZip magic w/ lax checksum handling
            decompressed = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(buffer.read())
            decompressed_buffer = BytesIO(decompressed)

            try:
                with tarfile.open(mode="r|", fileobj=decompressed_buffer) as tar:
                    try:
                        tar.getmembers()  # type: ignore[attr-defined]
                        return [m.name for m in tar.members]
                    except tarfile.ReadError as e:
                        if "unexpected end of data" in str(e):
                            return [m.name for m in tar.members]
            except tarfile.ReadError:
                continue

        return files

    @tool_method(
        name="list_tags",
        description="List available tags for a repo, e.g. repo='dotnet/runtime'.",
    )
    def list_tags(self, repo: str) -> list[str]:
        data = self._skopeo_json(f"list-tags docker://{self.registry}/{repo}")
        return data.get("Tags", [])

    @tool_method(
        name="get_manifest",
        description="Get manifest (skopeo inspect) for repo[:tag]. Defaults to latest (last tag).",
    )
    def get_manifest(self, repo: str, tag: str | None = None) -> dict[str, t.Any]:
        if not tag:
            tags = self.list_tags(repo)
            if not tags:
                raise ValueError("No tags found")
            tag = tags[-1]
            manifest = self._skopeo_json(f"inspect docker://{self.registry}/{repo}:{tag}")

            # Save manifest to default output directory
            Path.mkdir(self.default_out_dir, exist_ok=True)

            # Save the manifest to a file
            save_path = Path(self.default_out_dir).joinpath(
                f"{repo.replace('/', '_')}_manifest.json"
            )
            with save_path.open("w") as f:
                json.dump(manifest, f, indent=2)

        return f"manifest saved to {save_path}"

    @tool_method(
        name="get_config",
        description="Get config (skopeo inspect --config) for repo[:tag]. Defaults to latest (last tag).",
    )
    def get_config(self, repo: str, tag: str | None = None) -> dict[str, t.Any]:
        if not tag:
            tags = self.list_tags(repo)
            if not tags:
                raise ValueError("No tags found")
            tag = tags[-1]
        return self._skopeo_json(f"inspect --config docker://{self.registry}/{repo}:{tag}")

    @tool_method(
        name="list_files_in_latest",
        description="Peek each layer of the LATEST tag and return a mapping of layer digest -> file paths.",
    )
    def list_files_in_latest(self, repo: str) -> dict[str, list[str]]:
        manifest = self.get_manifest(repo)
        layers = manifest.get("Layers", [])
        out: dict[str, list[str]] = {}
        for digest in layers:
            out[digest] = self._peek_docker_layer(repo, digest)
        return out

    @tool_method(
        name="download_latest_layers",
        description="Download all layers for the LATEST tag into an output directory (extracts tar.gz layers).",
    )
    def download_latest_layers(self, repo: str, out_dir: str | None = None) -> str:
        out_dir = out_dir or self.default_out_dir
        Path.mkdir(out_dir, exist_ok=True)

        manifest = self.get_manifest(repo)
        layers = manifest.get("Layers", [])
        if not layers:
            return f"No layers found for {repo}"

        for digest in layers:
            url = f"https://{self.registry}/v2/{repo}/blobs/{digest}"
            try:
                with httpx.get(url, stream=True) as r:
                    r.raise_for_status()
                    with tarfile.open(fileobj=BytesIO(r.content)) as tar:
                        tar.extractall(out_dir, filter="data")
            except (httpx.HTTPError, tarfile.TarError, OSError) as e:
                return f"Failed on {digest}: {e}. Extracted so far to {out_dir}"

        return f"Extracted {len(layers)} layer(s) to {out_dir}"
