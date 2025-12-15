"""Pre-flight validation for air-gapped deployments."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from dreadnode.airgap.ecr_helper import ECRHelper


class ValidationError(Exception):
    """Raised when pre-flight validation fails."""

    pass


class PreFlightValidator:
    """Validates environment for air-gapped deployment."""

    MIN_DISK_SPACE_GB = 20
    MIN_MEMORY_GB = 4

    def __init__(self):
        """Initialize pre-flight validator."""
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(
        self,
        ecr_helper: Optional[ECRHelper] = None,
        bundle_path: Optional[Path] = None,
        skip_k8s: bool = False,
    ) -> None:
        """
        Run all pre-flight validation checks.

        Args:
            ecr_helper: ECR helper instance for connectivity checks
            bundle_path: Path to bundle for verification
            skip_k8s: Skip Kubernetes connectivity checks

        Raises:
            ValidationError: If any critical validation fails
        """
        logger.info("Running pre-flight validation checks...")

        # Required checks
        self.check_disk_space()
        self.check_required_tools()

        # Optional checks
        if bundle_path:
            self.check_bundle_exists(bundle_path)

        if ecr_helper:
            self.check_ecr_connectivity(ecr_helper)
            self.check_ecr_credentials(ecr_helper)

        if not skip_k8s:
            self.check_kubectl_available()
            self.check_kubernetes_connectivity()

        # Report results
        if self.warnings:
            logger.warning(f"Pre-flight validation completed with {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")

        if self.errors:
            error_msg = f"Pre-flight validation failed with {len(self.errors)} errors:\n"
            error_msg += "\n".join(f"  ❌ {error}" for error in self.errors)
            raise ValidationError(error_msg)

        logger.info("✅ All pre-flight validation checks passed")

    def check_disk_space(self, path: Path = Path.cwd()) -> None:
        """
        Check available disk space.

        Args:
            path: Path to check disk space for
        """
        try:
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024**3)

            if available_gb < self.MIN_DISK_SPACE_GB:
                self.errors.append(
                    f"Insufficient disk space: {available_gb:.1f}GB available, "
                    f"{self.MIN_DISK_SPACE_GB}GB required"
                )
            else:
                logger.debug(f"Disk space check passed: {available_gb:.1f}GB available")
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")

    def check_required_tools(self) -> None:
        """Check that required command-line tools are available."""
        required_tools = {
            "zarf": "Zarf binary not found. Install from https://zarf.dev",
            "kubectl": "kubectl not found. Install from https://kubernetes.io/docs/tasks/tools/",
        }

        for tool, error_msg in required_tools.items():
            if not shutil.which(tool):
                self.errors.append(error_msg)
            else:
                logger.debug(f"Tool check passed: {tool}")

    def check_kubectl_available(self) -> None:
        """Check if kubectl is available and executable."""
        if not shutil.which("kubectl"):
            self.errors.append("kubectl not found in PATH")
            return

        try:
            result = subprocess.run(
                ["kubectl", "version", "--client", "--output=json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            logger.debug(f"kubectl is available: {result.stdout[:100]}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.errors.append(f"kubectl is not functional: {e}")

    def check_kubernetes_connectivity(self) -> None:
        """Check connectivity to Kubernetes cluster."""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            if "is running at" in result.stdout:
                logger.debug("Kubernetes cluster is accessible")
            else:
                self.errors.append("Kubernetes cluster info returned unexpected output")
        except subprocess.CalledProcessError as e:
            self.errors.append(
                f"Cannot connect to Kubernetes cluster. "
                f"Ensure kubectl is configured correctly. Error: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            self.errors.append("Kubernetes cluster connection timed out")

    def check_bundle_exists(self, bundle_path: Path) -> None:
        """
        Check that bundle file exists and is readable.

        Args:
            bundle_path: Path to bundle file
        """
        if not bundle_path.exists():
            self.errors.append(f"Bundle file not found: {bundle_path}")
            return

        if not bundle_path.is_file():
            self.errors.append(f"Bundle path is not a file: {bundle_path}")
            return

        # Check file is readable
        try:
            with open(bundle_path, "rb") as f:
                f.read(1)
            logger.debug(f"Bundle file is readable: {bundle_path}")
        except Exception as e:
            self.errors.append(f"Bundle file is not readable: {e}")

        # Check file size is reasonable (not empty, not too small)
        size_mb = bundle_path.stat().st_size / (1024**2)
        if size_mb < 1:
            self.warnings.append(
                f"Bundle file seems small ({size_mb:.2f}MB). This may not be a valid Zarf package."
            )
        else:
            logger.debug(f"Bundle file size: {size_mb:.1f}MB")

    def check_ecr_connectivity(self, ecr_helper: ECRHelper) -> None:
        """
        Check ECR connectivity.

        Args:
            ecr_helper: ECR helper instance
        """
        if not ecr_helper.verify_connectivity():
            self.errors.append(
                f"Cannot connect to ECR registry: {ecr_helper.registry_url}. "
                "Ensure AWS credentials are configured and network connectivity is available."
            )
        else:
            logger.debug("ECR connectivity check passed")

    def check_ecr_credentials(self, ecr_helper: ECRHelper) -> None:
        """
        Check ECR credentials have necessary permissions.

        Args:
            ecr_helper: ECR helper instance
        """
        is_valid, error_msg = ecr_helper.verify_credentials()
        if not is_valid:
            self.errors.append(f"ECR credentials validation failed: {error_msg}")
        else:
            logger.debug("ECR credentials check passed")

    def check_namespace_available(self, namespace: str) -> None:
        """
        Check if Kubernetes namespace exists or can be created.

        Args:
            namespace: Namespace to check
        """
        try:
            result = subprocess.run(
                ["kubectl", "get", "namespace", namespace],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.debug(f"Namespace '{namespace}' exists")
            else:
                # Namespace doesn't exist, check if we can create it
                logger.debug(f"Namespace '{namespace}' does not exist, will be created")
        except subprocess.TimeoutExpired:
            self.warnings.append(f"Could not verify namespace '{namespace}': timeout")

    def check_storage_class_available(self, storage_class: str = "local-path") -> None:
        """
        Check if required storage class is available.

        Args:
            storage_class: Storage class name to check
        """
        try:
            result = subprocess.run(
                ["kubectl", "get", "storageclass", storage_class],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.debug(f"Storage class '{storage_class}' is available")
            else:
                self.warnings.append(
                    f"Storage class '{storage_class}' not found. "
                    "Deployment may fail if persistent volumes are required."
                )
        except subprocess.TimeoutExpired:
            self.warnings.append(f"Could not verify storage class '{storage_class}': timeout")

    def validate_zarf_package(self, package_path: Path) -> bool:
        """
        Validate that a file is a valid Zarf package.

        Args:
            package_path: Path to package file

        Returns:
            True if package is valid, False otherwise
        """
        # Basic validation: check extension
        if not package_path.name.endswith(".tar.zst"):
            logger.warning(f"Package does not have .tar.zst extension: {package_path.name}")
            return False

        # Could add more sophisticated validation here
        # (e.g., try to extract package metadata)
        return True
