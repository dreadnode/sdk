"""Orchestrates air-gapped installation workflow."""

from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dreadnode.airgap.ecr_helper import ECRHelper
from dreadnode.airgap.health import HealthChecker
from dreadnode.airgap.validator import PreFlightValidator, ValidationError
from dreadnode.airgap.zarf_wrapper import ZarfWrapper

console = Console()


class AirGapInstaller:
    """Orchestrates complete air-gapped installation workflow."""

    def __init__(
        self,
        bundle_path: Path,
        ecr_registry: str,
        namespace: str = "dreadnode",
        region: str | None = None,
    ):
        """
        Initialize air-gap installer.

        Args:
            bundle_path: Path to Zarf package bundle
            ecr_registry: ECR registry URL (e.g., 123456.dkr.ecr.us-east-1.amazonaws.com)
            namespace: Kubernetes namespace for deployment
            region: AWS region (auto-detected from registry if not provided)
        """
        self.bundle_path = bundle_path
        self.ecr_registry = ecr_registry
        self.namespace = namespace

        # Initialize components
        self.zarf = ZarfWrapper()
        self.ecr = ECRHelper(ecr_registry, region)
        self.validator = PreFlightValidator()
        self.health = HealthChecker(namespace)

    def install(
        self,
        *,
        skip_preflight: bool = False,
        skip_health_check: bool = False,
        components: list[str] | None = None,
    ) -> None:
        """
        Execute complete installation workflow.

        Args:
            skip_preflight: Skip pre-flight validation checks
            skip_health_check: Skip post-install health verification
            components: Specific Zarf components to deploy (default: all)

        Raises:
            ValidationError: If pre-flight validation fails
            RuntimeError: If installation fails
        """
        console.print("\n[bold cyan]🚀 Starting Dreadnode Air-Gapped Installation[/bold cyan]\n")

        try:
            # Phase 1: Pre-flight validation
            if not skip_preflight:
                self._run_preflight_checks()
            else:
                logger.warning("Skipping pre-flight checks (not recommended)")

            # Phase 2: Extract package metadata
            self._inspect_package()

            # Phase 3: Ensure ECR repositories
            self._prepare_ecr_repositories()

            # Phase 4: Deploy with Zarf
            self._deploy_package(components)

            # Phase 5: Health verification
            if not skip_health_check:
                self._verify_deployment()
            else:
                logger.warning("Skipping health checks")

            # Phase 6: Display summary
            self._display_completion_summary()

        except ValidationError as e:
            console.print(f"\n[bold red]❌ Pre-flight validation failed:[/bold red]\n{e}\n")
            raise
        except Exception as e:
            console.print(f"\n[bold red]❌ Installation failed:[/bold red] {e}\n")
            raise

    def _run_preflight_checks(self) -> None:
        """Run pre-flight validation checks."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running pre-flight checks...", total=None)

            try:
                self.validator.validate_all(
                    ecr_helper=self.ecr,
                    bundle_path=self.bundle_path,
                    skip_k8s=False,
                )
                progress.update(task, description="✅ Pre-flight checks passed")
            except ValidationError:
                progress.update(task, description="❌ Pre-flight checks failed")
                raise

    def _inspect_package(self) -> None:
        """Inspect Zarf package and extract metadata."""
        console.print("\n[bold]📦 Inspecting package...[/bold]")

        try:
            self.metadata = self.zarf.inspect_package(self.bundle_path)

            # Extract image list for ECR preparation
            self.images = self._extract_images_from_metadata(self.metadata)

            console.print(f"  Package: [cyan]{self.metadata.get('name', 'unknown')}[/cyan]")
            console.print(f"  Version: [cyan]{self.metadata.get('version', 'unknown')}[/cyan]")
            console.print(f"  Components: [cyan]{len(self.metadata.get('components', []))}[/cyan]")
            console.print(f"  Images: [cyan]{len(self.images)}[/cyan]")

        except Exception as e:
            console.print(f"[red]Failed to inspect package: {e}[/red]")
            raise RuntimeError(f"Package inspection failed: {e}") from e

    def _extract_images_from_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """
        Extract list of images from Zarf package metadata.

        Args:
            metadata: Zarf package metadata

        Returns:
            List of image names
        """
        images = []

        for component in metadata.get("components", []):
            # Images can be in different places depending on component type
            component_images = component.get("images", [])
            images.extend(component_images)

            # Also check charts for images
            for chart in component.get("charts", []):
                chart_images = chart.get("images", [])
                images.extend(chart_images)

        # Deduplicate
        return list(set(images))

    def _prepare_ecr_repositories(self) -> None:
        """Ensure ECR repositories exist for all images."""
        if not self.images:
            logger.info("No images to prepare ECR repositories for")
            return

        console.print(
            f"\n[bold]🏗️  Preparing ECR repositories for {len(self.images)} images...[/bold]"
        )

        try:
            created_repos = self.ecr.ensure_repositories(self.images)

            if created_repos:
                console.print(f"  Created [green]{len(created_repos)}[/green] new repositories")
            else:
                console.print("  All repositories already exist")

        except Exception as e:
            console.print(f"[red]Failed to prepare ECR repositories: {e}[/red]")
            raise RuntimeError(f"ECR preparation failed: {e}") from e

    def _deploy_package(self, components: list[str] | None = None) -> None:
        """
        Deploy Zarf package to Kubernetes cluster.

        Args:
            components: Specific components to deploy
        """
        console.print("\n[bold]🚀 Deploying platform components...[/bold]")
        console.print(
            "  This may take several minutes depending on package size and cluster speed.\n"
        )

        try:
            # Set registry override to point to customer ECR
            set_variables = {
                "REGISTRY": self.ecr_registry,
                "NAMESPACE": self.namespace,
            }

            self.zarf.deploy_package(
                package_path=self.bundle_path,
                components=components,
                set_variables=set_variables,
                confirm=True,
            )

            console.print("\n  [green]✅ Package deployed successfully[/green]")

        except Exception as e:
            console.print(f"\n  [red]❌ Deployment failed: {e}[/red]")
            raise RuntimeError(f"Package deployment failed: {e}") from e

    def _verify_deployment(self) -> None:
        """Verify deployment health."""
        console.print("\n[bold]🏥 Verifying deployment health...[/bold]")

        try:
            # Wait for pods to be ready
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Waiting for pods to become ready...", total=None)

                try:
                    self.health.wait_for_ready(timeout=600)
                    progress.update(task, description="✅ All pods are ready")
                except Exception as e:
                    progress.update(task, description=f"❌ Pod readiness check failed: {e}")
                    raise

            # Verify API health
            try:
                self.health.verify_api_health()
                console.print("  ✅ Platform API is healthy")
            except Exception as e:  # noqa: BLE001
                console.print(f"  ⚠️  API health check warning: {e}")

        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Warning: Health verification had issues: {e}[/yellow]")
            console.print(
                "[yellow]The platform may still be functional. "
                "Check pod logs for more details.[/yellow]"
            )

    def _display_completion_summary(self) -> None:
        """Display installation completion summary."""
        console.print("\n" + "=" * 70)
        console.print("[bold green]✅ Installation Complete![/bold green]")
        console.print("=" * 70 + "\n")

        # Get deployment summary
        try:
            summary = self.health.get_deployment_summary()

            console.print("[bold]Deployed Resources:[/bold]")
            console.print(f"  Namespace: [cyan]{summary['namespace']}[/cyan]")
            console.print(f"  Pods: [cyan]{summary['pods']['total']}[/cyan]")
            console.print(f"  Services: [cyan]{summary['services']['total']}[/cyan]")
            console.print(f"  Deployments: [cyan]{summary['deployments']['total']}[/cyan]")

            # Check PVCs
            pv_status = self.health.check_persistent_volumes()
            if pv_status["total"] > 0:
                console.print(
                    f"  Persistent Volume Claims: [cyan]{pv_status['bound']}/{pv_status['total']}[/cyan] bound"
                )

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not get complete summary: {e}")

        console.print("\n[bold]Next Steps:[/bold]")
        console.print(f"  1. Verify deployment: [cyan]kubectl get pods -n {self.namespace}[/cyan]")
        console.print(f"  2. Check services: [cyan]kubectl get services -n {self.namespace}[/cyan]")
        console.print(f"  3. View logs: [cyan]kubectl logs -n {self.namespace} <pod-name>[/cyan]")
        console.print(
            f"  4. Access platform: [cyan]kubectl port-forward -n {self.namespace} svc/<service-name> 8080:80[/cyan]"
        )
        console.print()
