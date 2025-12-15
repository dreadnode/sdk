"""Wrapper for Zarf CLI operations."""

import json
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from loguru import logger


class ZarfWrapper:
    """Manages Zarf CLI interactions for air-gapped deployments."""

    def __init__(self, zarf_binary: str = "zarf"):
        """
        Initialize Zarf wrapper.

        Args:
            zarf_binary: Path to zarf binary (default: "zarf" from PATH)

        Raises:
            RuntimeError: If Zarf binary is not found or not executable
        """
        self.zarf_binary = zarf_binary
        self._verify_zarf_available()

    def _verify_zarf_available(self) -> None:
        """Verify that Zarf binary is available and executable."""
        if not shutil.which(self.zarf_binary):
            raise RuntimeError(
                f"Zarf binary '{self.zarf_binary}' not found in PATH. "
                "Please install Zarf from https://zarf.dev or specify the full path."
            )

        # Verify Zarf is executable and get version
        try:
            result = subprocess.run(  # nosec B603, B607
                [self.zarf_binary, "version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            logger.debug(f"Using Zarf: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute Zarf: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Zarf command timed out") from e

    def create_package(
        self,
        source_dir: Path,
        output_dir: Path,
        version: str | None = None,
        *,
        confirm: bool = True,
        set_variables: dict[str, str] | None = None,
    ) -> Path:
        """
        Create Zarf package from source directory.

        Args:
            source_dir: Directory containing zarf.yaml
            output_dir: Directory where package will be created
            version: Platform version to package
            confirm: Auto-confirm package creation (no interactive prompts)
            set_variables: Additional Zarf variables to set

        Returns:
            Path to created package file

        Raises:
            RuntimeError: If package creation fails
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.zarf_binary,
            "package",
            "create",
            str(source_dir),
            f"--output-directory={output_dir}",
        ]

        if version:
            cmd.append(f"--set=version={version}")

        if set_variables:
            for key, value in set_variables.items():
                cmd.append(f"--set={key}={value}")

        if confirm:
            cmd.append("--confirm")

        logger.info(f"Creating Zarf package from {source_dir}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # nosec B603, B607
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,  # 10 minute timeout for package creation
            )
            logger.debug(f"Zarf output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Zarf package creation failed: {e.stderr}")
            raise RuntimeError(f"Zarf package creation failed: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Zarf package creation timed out after 10 minutes") from e

        # Find created package
        return self._find_package(output_dir)

    def deploy_package(
        self,
        package_path: Path,
        components: list[str] | None = None,
        set_variables: dict[str, str] | None = None,
        *,
        confirm: bool = True,
    ) -> None:
        """
        Deploy Zarf package to Kubernetes cluster.

        Args:
            package_path: Path to Zarf package file
            components: List of components to deploy (default: all)
            set_variables: Variables to set during deployment
            confirm: Auto-confirm deployment (no interactive prompts)

        Raises:
            RuntimeError: If package deployment fails
        """
        if not package_path.exists():
            raise RuntimeError(f"Package not found: {package_path}")

        cmd = [self.zarf_binary, "package", "deploy", str(package_path)]

        if components:
            cmd.extend(f"--components={component}" for component in components)

        if set_variables:
            for key, value in set_variables.items():
                cmd.append(f"--set={key}={value}")

        if confirm:
            cmd.append("--confirm")

        logger.info(f"Deploying Zarf package {package_path.name}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # nosec B603, B607
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minute timeout for deployment
            )
            logger.debug(f"Zarf output: {result.stdout}")
            logger.info("Package deployed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Zarf package deployment failed: {e.stderr}")
            raise RuntimeError(f"Zarf package deployment failed: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Zarf package deployment timed out after 30 minutes") from e

    def inspect_package(
        self,
        package_path: Path,
        sbom_output: Path | None = None,
    ) -> dict[str, Any]:
        """
        Inspect Zarf package and extract metadata.

        Args:
            package_path: Path to Zarf package file
            sbom_output: Directory to extract SBOMs to (optional)

        Returns:
            Package metadata as dictionary

        Raises:
            RuntimeError: If package inspection fails
        """
        if not package_path.exists():
            raise RuntimeError(f"Package not found: {package_path}")

        cmd = [
            self.zarf_binary,
            "package",
            "inspect",
            str(package_path),
            "--output=json",
        ]

        if sbom_output:
            sbom_output.mkdir(parents=True, exist_ok=True)
            cmd.append(f"--sbom-out={sbom_output}")

        logger.debug(f"Inspecting package: {package_path.name}")

        try:
            result = subprocess.run(  # nosec B603, B607
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return json.loads(result.stdout)  # type: ignore[no-any-return]
        except subprocess.CalledProcessError as e:
            logger.error(f"Zarf package inspection failed: {e.stderr}")
            raise RuntimeError(f"Zarf package inspection failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Zarf output: {e}")
            raise RuntimeError(f"Failed to parse Zarf output: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Zarf package inspection timed out") from e

    def list_packages(self, directory: Path) -> list[Path]:
        """
        List all Zarf packages in a directory.

        Args:
            directory: Directory to search for packages

        Returns:
            List of paths to Zarf package files
        """
        if not directory.exists():
            return []

        # Zarf packages typically have .tar.zst extension
        return sorted(directory.glob("*.tar.zst"))

    def _find_package(self, output_dir: Path) -> Path:
        """
        Find the most recently created Zarf package in output directory.

        Args:
            output_dir: Directory to search

        Returns:
            Path to most recent package

        Raises:
            RuntimeError: If no package found
        """
        packages = self.list_packages(output_dir)
        if not packages:
            raise RuntimeError(f"No Zarf package found in {output_dir}")

        # Return most recent package (by modification time)
        return max(packages, key=lambda p: p.stat().st_mtime)

    def init_cluster(
        self,
        components: list[str] | None = None,
        storage_class: str = "local-path",
        *,
        confirm: bool = True,
    ) -> None:
        """
        Initialize Zarf in the Kubernetes cluster.

        Args:
            components: Components to initialize (e.g., ["git-server"])
            storage_class: Storage class for Zarf components
            confirm: Auto-confirm initialization

        Raises:
            RuntimeError: If initialization fails
        """
        cmd = [
            self.zarf_binary,
            "init",
            f"--storage-class={storage_class}",
        ]

        if components:
            cmd.extend(f"--components={component}" for component in components)

        if confirm:
            cmd.append("--confirm")

        logger.info("Initializing Zarf in cluster...")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # nosec B603, B607
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
            logger.debug(f"Zarf output: {result.stdout}")
            logger.info("Zarf initialized successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Zarf initialization failed: {e.stderr}")
            raise RuntimeError(f"Zarf initialization failed: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("Zarf initialization timed out") from e
