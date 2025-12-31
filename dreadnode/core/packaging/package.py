from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import tomlkit
import tomllib
from jinja2 import BaseLoader, Environment

from dreadnode.core.packaging.manifest import MANIFEST_TYPES

if TYPE_CHECKING:
    from dreadnode.core.storage.storage import Storage

PackageType = Literal["datasets", "models", "toolsets", "agents", "environments"]

CAS_TYPES = {"dataset", "model"}

ARTIFACT_EXTENSIONS = {
    ".parquet",
    ".csv",
    ".arrow",
    ".feather",
    ".jsonl",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".h5",
    ".pb",
    ".tar",
    ".gz",
    ".zip",
}


@dataclass
class BuildResult:
    success: bool
    wheel: Path | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class PushResult:
    success: bool
    wheel: Path | None = None
    blobs_uploaded: int = 0
    blobs_skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PullResult:
    success: bool
    errors: list[str] = field(default_factory=list)


class _ResourceLoader(BaseLoader):
    """Jinja loader for importlib.resources."""

    def __init__(self, package: str):
        self.root = files(package)

    def get_source(self, environment: Environment, template: str):
        path = self.root.joinpath(template)
        source = path.read_text()
        return source, template, lambda: True


def _normalize_module_name(name: str) -> str:
    """Convert component name to valid Python module name."""
    normalized = re.sub(r"[.\-/]", "_", name)
    normalized = re.sub(r"[^a-zA-Z0-9_]", "", normalized)
    if normalized and normalized[0].isdigit():
        normalized = "_" + normalized
    return normalized.lower()


def _normalize_entry_point_name(name: str) -> str:
    """Convert to valid entry point name."""
    return name.replace("/", ".")


def _parse_package_uri(uri: str) -> str:
    """Parse a package URI and return the pip package name.

    Supports URI formats:
        - dataset://org/name -> org.name (org/name style)
        - dataset://unique_id -> unique_id (unique ID style)
        - agent://org/name -> org.name
        - model://unique_id -> unique_id
        - etc.

    Plain package names are returned as-is for backwards compatibility.

    Args:
        uri: Package URI or plain name.

    Returns:
        The pip package name.
    """
    # Check for URI scheme (e.g., "dataset://")
    if "://" in uri:
        scheme, path = uri.split("://", 1)

        # Validate scheme is a known package type
        valid_schemes = {"dataset", "agent", "model", "toolset", "environment"}
        if scheme not in valid_schemes:
            raise ValueError(
                f"Unknown package type: {scheme}. "
                f"Valid types: {', '.join(sorted(valid_schemes))}"
            )

        # Check if path contains "/" (org/name style) or not (unique ID style)
        if "/" in path:
            # org/name -> org.name (dot notation for pip)
            return path.replace("/", ".")
        else:
            # unique_id -> return as-is
            return path

    # Plain package name - return as-is
    return uri


def _collect_artifacts(module_dir: Path) -> list[Path]:
    """Collect artifact files that should go to CAS."""
    return [
        p for p in module_dir.rglob("*") if p.is_file() and p.suffix.lower() in ARTIFACT_EXTENSIONS
    ]


def _get_template_type(package_type: PackageType) -> str:
    """Convert package type to template directory name (singular)."""
    return package_type.rstrip("s")


def _render_templates(
    package_type: PackageType,
    dest_dir: Path,
    module_name: str,
    context: dict[str, str],
) -> None:
    """Render Jinja templates from package resources."""
    template_type = _get_template_type(package_type)
    package = f"dreadnode.core.packaging.templates.{template_type}"
    root = files(package)
    env = Environment(loader=_ResourceLoader(package))

    def process_path(rel_path: str) -> str:
        result = rel_path.replace("{{ package_name }}", module_name)
        result = result.replace("pkg", module_name)
        return result.removesuffix(".j2")

    def walk_resources(resource, rel_path: str = ""):
        for item in resource.iterdir():
            if item.name in ("__init__.py", "__pycache__"):
                continue

            item_rel = f"{rel_path}/{item.name}" if rel_path else item.name

            if item.is_file():
                dest_rel = process_path(item_rel)
                dest_path = dest_dir / dest_rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if item.name.endswith(".j2"):
                    template = env.get_template(item_rel)
                    dest_path.write_text(template.render(**context))
                else:
                    dest_path.write_bytes(item.read_bytes())

            elif hasattr(item, "iterdir"):
                walk_resources(item, item_rel)

    for item in root.iterdir():
        if item.name not in ("__init__.py", "__pycache__"):
            if item.is_file():
                dest_rel = process_path(item.name)
                dest_path = dest_dir / dest_rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if item.name.endswith(".j2"):
                    template = env.get_template(item.name)
                    dest_path.write_text(template.render(**context))
                else:
                    dest_path.write_bytes(item.read_bytes())
            elif hasattr(item, "iterdir"):
                walk_resources(item, item.name)


class Package:
    """A dreadnode component package."""

    def __init__(self, path: Path | None = None):
        """Load an existing package.

        Args:
            path: Project directory. Defaults to current directory.
        """
        self.project_dir = (path or Path.cwd()).resolve()
        self._load_info()

    def _load_info(self) -> None:
        """Load project info from pyproject.toml."""
        pyproject = self.project_dir / "pyproject.toml"
        if not pyproject.exists():
            raise FileNotFoundError(f"No pyproject.toml found in {self.project_dir}")

        with open(pyproject, "rb") as f:
            config = tomllib.load(f)

        self.version = config["project"]["version"]
        self.description = config["project"].get("description", "")

        # Find dreadnode entry point
        entry_points = config.get("project", {}).get("entry-points", {})
        for group, eps in entry_points.items():
            if group.startswith("dreadnode."):
                self.entry_point_name = next(iter(eps.keys()))
                self.module_name = next(iter(eps.values()))
                self.package_type = group.replace("dreadnode.", "").rstrip("s")
                self.org = self.entry_point_name.split(".")[0]
                self.name = (
                    self.entry_point_name.split(".", 1)[1]
                    if "." in self.entry_point_name
                    else self.entry_point_name
                )
                break
        else:
            raise ValueError("No dreadnode entry points found in pyproject.toml")

        self.module_dir = self.project_dir / "src" / self.module_name
        self.manifest_path = self.module_dir / "manifest.json"

    @property
    def manifest(self):
        """Load and return the manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError("manifest.json not found")
        manifest_class = MANIFEST_TYPES[self.package_type]
        return manifest_class.model_validate_json(self.manifest_path.read_text())

    @classmethod
    def init(
        cls,
        name: str,
        package_type: PackageType,
        storage: "Storage",
    ) -> "Package":
        """Initialize a new package.

        Args:
            name: package name (e.g., "sentiment-data").
            package_type: Type of component.
            storage: Storage instance for determining package location.

        Returns:
            The initialized Package.
        """
        module_name = _normalize_module_name(name)
        entry_point_name = _normalize_entry_point_name(name)
        package_name = name.replace("/", "-").replace(".", "-")

        # Get the package source directory from storage
        project_dir = storage.package_source_path(package_type, name)

        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv not found")

        subprocess.run(
            [uv, "init", "--package", "--name", package_name, str(project_dir)],
            check=True,
            capture_output=True,
        )

        pyproject_path = project_dir / "pyproject.toml"
        with open(pyproject_path) as f:
            config = tomlkit.load(f)

        config["project"]["description"] = f"A dreadnode {package_type}"

        deps = config["project"].get("dependencies", [])
        if "dreadnode" not in deps:
            deps.append("dreadnode")
        config["project"]["dependencies"] = deps

        entry_points = config["project"].get("entry-points", tomlkit.table())
        group = f"dreadnode.{package_type}s"
        if group not in entry_points:
            entry_points[group] = tomlkit.table()
        entry_points[group][entry_point_name] = module_name
        config["project"]["entry-points"] = entry_points

        if "scripts" in config["project"]:
            del config["project"]["scripts"]

        with open(pyproject_path, "w") as f:
            tomlkit.dump(config, f)

        src_dir = project_dir / "src" / module_name
        for default_file in ["__init__.py", "main.py"]:
            default_path = src_dir / default_file
            if default_path.exists():
                default_path.unlink()

        context = {
            "name": name,
            "module_name": module_name,
            "entry_point_name": entry_point_name,
            "package_name": package_name,
            "description": f"A dreadnode {package_type}",
        }

        _render_templates(package_type, project_dir, module_name, context)

        # Use singular form for manifest lookup
        singular_type = _get_template_type(package_type)
        manifest = MANIFEST_TYPES[singular_type]()

        manifest_path = src_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))

        return cls(project_dir)

    def build(self) -> BuildResult:
        """Build the package wheel.

        Returns:
            BuildResult with success status and wheel path.
        """
        result = BuildResult(success=False)

        uv = shutil.which("uv")
        if not uv:
            result.errors.append("uv not found")
            return result

        try:
            subprocess.run(
                [uv, "build", "--wheel"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            result.errors.append(f"Build failed: {e.stderr}")
            return result

        wheels = list((self.project_dir / "dist").glob("*.whl"))
        if wheels:
            result.wheel = wheels[0]
            result.success = True

        return result

    def push(self, storage: Storage, *, skip_upload: bool = False) -> PushResult:
        """Push the package to registry.

        For datasets and models, uploads artifacts to CAS first.

        Args:
            storage: Storage instance for the org.
            skip_upload: Skip uploading to remote (local only).

        Returns:
            PushResult with status and details.
        """
        result = PushResult(success=False)

        if not self.module_dir.exists():
            result.errors.append(f"Module directory not found: {self.module_dir}")
            return result

        if not self.manifest_path.exists():
            result.errors.append("manifest.json not found")
            return result

        manifest = self.manifest

        # Handle CAS artifacts for datasets and models
        if self.package_type in CAS_TYPES:
            artifacts = _collect_artifacts(self.module_dir)

            if artifacts:
                # Batch hash all files
                hashes = storage.hash_files(artifacts)

                # Build oid mapping
                files_to_upload: dict[Path, str] = {}
                for file_path in artifacts:
                    rel_path = str(file_path.relative_to(self.module_dir))
                    file_hash = hashes[file_path]
                    oid = f"sha256:{file_hash}"
                    manifest.artifacts[rel_path] = oid

                    # Store locally
                    if not storage.blob_exists(oid):
                        storage.store_blob(oid, file_path)

                    files_to_upload[file_path] = oid

                # Batch upload to remote
                if not skip_upload:
                    uploaded, skipped = storage.upload_blobs(files_to_upload)
                    result.blobs_uploaded = uploaded
                    result.blobs_skipped = skipped
                else:
                    result.blobs_uploaded = len(files_to_upload)

                # Compute artifacts hash
                artifacts_json = json.dumps(manifest.artifacts, sort_keys=True)
                manifest.artifacts_hash = (
                    f"sha256:{hashlib.sha256(artifacts_json.encode()).hexdigest()}"
                )

                # Write updated manifest
                self.manifest_path.write_text(manifest.model_dump_json(indent=2))

                # Store manifest in local storage
                storage.store_manifest(
                    package_type=f"{self.package_type}s",
                    name=self.name,
                    version=self.version,
                    content=manifest.model_dump_json(indent=2),
                )

                # Remove artifacts from module
                for file_path in artifacts:
                    file_path.unlink()
                    parent = file_path.parent
                    while parent != self.module_dir and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent

        # Build wheel
        build_result = self.build()
        if not build_result.success:
            result.errors.extend(build_result.errors)
            return result

        result.wheel = build_result.wheel

        # Upload wheel to PyPI registry
        if not skip_upload and result.wheel:
            if storage.session is None:
                result.errors.append("No API session configured for upload")
                return result

            try:
                storage.session.api.upload_pypi_package(
                    name=self.entry_point_name,
                    version=self.version,
                    wheel_path=result.wheel,
                )
            except Exception as e:
                result.errors.append(f"Failed to upload wheel: {e}")
                return result

        result.success = True
        return result

    @staticmethod
    def pull(
        *packages: str,
        upgrade: bool = False,
        _storage: "Storage",
    ) -> PullResult:
        """Install packages from registry.

        Packages can be specified using URI syntax:
            - dataset://org/name - Install a dataset package
            - agent://org/name - Install an agent package
            - model://org/name - Install a model package
            - toolset://org/name - Install a toolset package
            - environment://org/name - Install an environment package

        Or as plain package names (for backwards compatibility):
            - org.name - Direct pip package name

        Args:
            packages: Package names or URIs to install.
            _storage: Storage instance with session for API access.
            upgrade: Upgrade if already installed.

        Returns:
            PullResult with status.
        """
        result = PullResult(success=False)

        if not packages:
            result.errors.append("No packages specified")
            return result

        if _storage.session is None:
            result.errors.append("No API session configured - cannot access registry")
            return result

        uv = shutil.which("uv")
        if not uv:
            result.errors.append("uv not found")
            return result

        # Parse package URIs and convert to pip package names
        pip_packages = []
        for pkg in packages:
            pip_name = _parse_package_uri(pkg)
            pip_packages.append(pip_name)

        # Get registry URL from API client
        registry_url = _storage.session.api.pypi_registry_url

        # Build uv pip install command with registry URL
        # Use --extra-index-url to also check PyPI for dependencies
        cmd = [uv, "pip", "install", "--extra-index-url", registry_url]

        if upgrade:
            cmd.append("--upgrade")

        cmd.extend(pip_packages)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            result.success = True
        except subprocess.CalledProcessError as e:
            result.errors.append(f"Install failed: {e.stderr}")

        return result
