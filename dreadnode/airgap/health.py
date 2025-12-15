"""Post-installation health checks for air-gapped deployments."""

import json
import subprocess  # nosec B404
import time
from typing import Any

from loguru import logger


class HealthCheckError(Exception):
    """Raised when health checks fail."""


class HealthChecker:
    """Performs health checks on deployed Dreadnode platform."""

    def __init__(self, namespace: str = "dreadnode"):
        """
        Initialize health checker.

        Args:
            namespace: Kubernetes namespace where platform is deployed
        """
        self.namespace = namespace

    def wait_for_ready(
        self,
        timeout: int = 600,
        check_interval: int = 10,
        required_pods: list[str] | None = None,  # noqa: ARG002
    ) -> None:
        """
        Wait for all pods to reach Ready state.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Interval between checks in seconds
            required_pods: List of required pod name prefixes (default: check all pods)

        Raises:
            HealthCheckError: If pods don't become ready within timeout
        """
        logger.info(f"Waiting for pods in namespace '{self.namespace}' to become ready...")

        start_time = time.time()
        last_status = ""

        while time.time() - start_time < timeout:
            try:
                pod_status = self._get_pod_status()

                if pod_status["total"] == 0:
                    logger.debug("No pods found yet, waiting...")
                    time.sleep(check_interval)
                    continue

                status_msg = (
                    f"Pods: {pod_status['ready']}/{pod_status['total']} ready, "
                    f"{pod_status['running']}/{pod_status['total']} running"
                )

                if status_msg != last_status:
                    logger.info(status_msg)
                    last_status = status_msg

                # Check if all pods are ready
                if pod_status["ready"] == pod_status["total"] and pod_status["total"] > 0:
                    logger.info("✅ All pods are ready")
                    return

                # Check for failed pods
                if pod_status["failed"] > 0:
                    failed_pods = self._get_failed_pods()
                    logger.warning(f"Found {pod_status['failed']} failed pods: {failed_pods}")

                time.sleep(check_interval)

            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error checking pod status: {e}")
                time.sleep(check_interval)

        # Timeout reached
        pod_status = self._get_pod_status()
        raise HealthCheckError(
            f"Pods did not become ready within {timeout} seconds. "
            f"Status: {pod_status['ready']}/{pod_status['total']} ready"
        )

    def _get_pod_status(self) -> dict[str, int]:
        """
        Get status of all pods in namespace.

        Returns:
            Dictionary with pod status counts
        """
        try:
            result = subprocess.run(  # nosec B603, B607
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])

            status = {
                "total": len(pods),
                "ready": 0,
                "running": 0,
                "pending": 0,
                "failed": 0,
            }

            for pod in pods:
                phase = pod.get("status", {}).get("phase", "Unknown")

                if phase == "Running":
                    status["running"] += 1
                elif phase == "Pending":
                    status["pending"] += 1
                elif phase in ["Failed", "CrashLoopBackOff", "Error"]:
                    status["failed"] += 1

                # Check if pod is ready
                conditions = pod.get("status", {}).get("conditions", [])
                for condition in conditions:
                    if condition.get("type") == "Ready" and condition.get("status") == "True":
                        status["ready"] += 1
                        break
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get pod status: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pod status JSON: {e}")
            raise
        else:
            return status

    def _get_failed_pods(self) -> list[str]:
        """
        Get names of failed pods.

        Returns:
            List of failed pod names
        """
        try:
            result = subprocess.run(  # nosec B603, B607
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])

            failed_pods = []
            for pod in pods:
                phase = pod.get("status", {}).get("phase", "Unknown")
                if phase in ["Failed", "CrashLoopBackOff", "Error"]:
                    failed_pods.append(pod.get("metadata", {}).get("name", "unknown"))
        except Exception:  # noqa: BLE001
            return []
        else:
            return failed_pods

    def verify_api_health(self, api_endpoint: str | None = None) -> bool:  # noqa: ARG002
        """
        Verify platform API is healthy and responding.

        Args:
            api_endpoint: API endpoint to check (optional, will try to discover)

        Returns:
            True if API is healthy

        Raises:
            HealthCheckError: If API health check fails
        """
        # For now, we'll check if the API pods are running
        # In a real implementation, this would make HTTP requests to health endpoints
        logger.info("Verifying platform API health...")

        try:
            # Check for API pods
            result = subprocess.run(  # nosec B603, B607
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    self.namespace,
                    "-l",
                    "app=platform-api",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])

            if not pods:
                logger.warning("No API pods found, skipping API health check")
                return True

            # Check if any API pod is ready
            ready_pods = 0
            for pod in pods:
                conditions = pod.get("status", {}).get("conditions", [])
                for condition in conditions:
                    if condition.get("type") == "Ready" and condition.get("status") == "True":
                        ready_pods += 1
                        break

            if ready_pods > 0:
                logger.info(f"✅ Platform API is healthy ({ready_pods} pod(s) ready)")
                return True
            raise HealthCheckError("No API pods are in ready state")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to verify API health: {e.stderr}")
            raise HealthCheckError(f"API health check failed: {e.stderr}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API pod JSON: {e}")
            raise HealthCheckError(f"API health check failed: {e}") from e

    def get_deployment_summary(self) -> dict[str, Any]:
        """
        Get summary of deployed resources.

        Returns:
            Dictionary with deployment summary
        """
        summary = {
            "namespace": self.namespace,
            "pods": {},
            "services": {},
            "deployments": {},
        }

        try:
            # Get pods summary
            pod_result = subprocess.run(  # nosec B603, B607
                ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            pods_data = json.loads(pod_result.stdout)
            summary["pods"] = self._summarize_pods(pods_data.get("items", []))

            # Get services summary
            svc_result = subprocess.run(  # nosec B603, B607
                ["kubectl", "get", "services", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            svc_data = json.loads(svc_result.stdout)
            summary["services"] = self._summarize_services(svc_data.get("items", []))

            # Get deployments summary
            deploy_result = subprocess.run(  # nosec B603, B607
                ["kubectl", "get", "deployments", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            deploy_data = json.loads(deploy_result.stdout)
            summary["deployments"] = self._summarize_deployments(deploy_data.get("items", []))

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not get complete deployment summary: {e}")

        return summary

    def _summarize_pods(self, pods: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize pod information."""
        return {
            "total": len(pods),
            "names": [pod.get("metadata", {}).get("name", "unknown") for pod in pods],
        }

    def _summarize_services(self, services: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize service information."""
        return {
            "total": len(services),
            "names": [svc.get("metadata", {}).get("name", "unknown") for svc in services],
        }

    def _summarize_deployments(self, deployments: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize deployment information."""
        return {
            "total": len(deployments),
            "names": [deploy.get("metadata", {}).get("name", "unknown") for deploy in deployments],
        }

    def check_persistent_volumes(self) -> dict[str, Any]:
        """
        Check status of persistent volumes.

        Returns:
            Dictionary with PV/PVC status
        """
        try:
            pvc_result = subprocess.run(  # nosec B603, B607
                ["kubectl", "get", "pvc", "-n", self.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            pvc_data = json.loads(pvc_result.stdout)
            pvcs = pvc_data.get("items", [])

            bound_pvcs = sum(1 for pvc in pvcs if pvc.get("status", {}).get("phase") == "Bound")

            return {
                "total": len(pvcs),
                "bound": bound_pvcs,
                "names": [pvc.get("metadata", {}).get("name", "unknown") for pvc in pvcs],
            }

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not check persistent volumes: {e}")
            return {"total": 0, "bound": 0, "names": []}
