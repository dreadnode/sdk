"""CLI commands for air-gapped deployment operations."""

import typing as t
from pathlib import Path

import cyclopts
from rich.table import Table

from dreadnode.airgap.ecr_helper import ECRHelper
from dreadnode.airgap.installer import AirGapInstaller
from dreadnode.airgap.validator import PreFlightValidator, ValidationError
from dreadnode.airgap.zarf_wrapper import ZarfWrapper
from dreadnode.logging_ import console, print_error, print_info, print_success

cli = cyclopts.App(name="airgap", help="Air-gapped deployment operations")


@cli.command
def package_create(
    version: t.Annotated[
        str,
        cyclopts.Parameter(help="Platform version to package"),
    ],
    source_dir: t.Annotated[
        Path,
        cyclopts.Parameter(help="Directory containing zarf.yaml (default: ./zarf)"),
    ] = Path("./zarf"),
    output_dir: t.Annotated[
        Path,
        cyclopts.Parameter(help="Output directory for package (default: ./packages)"),
    ] = Path("./packages"),
) -> None:
    """
    Create air-gapped deployment package.

    This command creates a Zarf package containing all platform components,
    container images, and dependencies for air-gapped deployment.
    """
    print_info(f"Creating air-gapped package for version {version}...")

    if not source_dir.exists():
        print_error(f"Source directory not found: {source_dir}")
        raise SystemExit(1)

    zarf_yaml = source_dir / "zarf.yaml"
    if not zarf_yaml.exists():
        print_error(f"zarf.yaml not found in {source_dir}")
        raise SystemExit(1)

    try:
        zarf = ZarfWrapper()

        console.print("\n[bold]Creating Zarf package...[/bold]")
        console.print("  This may take several minutes to download and bundle all images.\n")

        package_path = zarf.create_package(
            source_dir=source_dir,
            output_dir=output_dir,
            version=version,
            confirm=True,
        )

        console.print()
        print_success(f"Package created: {package_path}")

        # Display package info
        console.print("\n[bold]Package Information:[/bold]")
        size_mb = package_path.stat().st_size / (1024**2)
        console.print(f"  Location: [cyan]{package_path}[/cyan]")
        console.print(f"  Size: [cyan]{size_mb:.1f} MB[/cyan]")

    except RuntimeError as e:
        print_error(f"Package creation failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@cli.command
def package_inspect(
    bundle: t.Annotated[
        Path,
        cyclopts.Parameter(help="Path to Zarf package bundle"),
    ],
    sbom_out: t.Annotated[
        Path | None,
        cyclopts.Parameter(help="Directory to extract SBOMs to"),
    ] = None,
) -> None:
    """
    Inspect air-gapped deployment package.

    Display package metadata, components, and optionally extract SBOMs.
    """
    if not bundle.exists():
        print_error(f"Bundle not found: {bundle}")
        raise SystemExit(1)

    try:
        zarf = ZarfWrapper()

        print_info("Inspecting package...")
        metadata = zarf.inspect_package(bundle, sbom_output=sbom_out)

        # Display package metadata
        console.print("\n[bold cyan]Package Metadata[/bold cyan]")
        console.print(f"  Name: [cyan]{metadata.get('name', 'unknown')}[/cyan]")
        console.print(f"  Version: [cyan]{metadata.get('version', 'unknown')}[/cyan]")
        console.print(f"  Description: {metadata.get('description', 'N/A')}")

        # Display components
        components = metadata.get("components", [])
        if components:
            console.print(f"\n[bold cyan]Components ({len(components)})[/bold cyan]")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Name")
            table.add_column("Required")
            table.add_column("Images")
            table.add_column("Charts")

            for component in components:
                name = component.get("name", "unknown")
                required = "Yes" if component.get("required", False) else "No"
                images = len(component.get("images", []))
                charts = len(component.get("charts", []))

                table.add_row(name, required, str(images), str(charts))

            console.print(table)

        # Display SBOM extraction info
        if sbom_out:
            console.print(f"\n[green]✅ SBOMs extracted to: {sbom_out}[/green]")

    except RuntimeError as e:
        print_error(f"Package inspection failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@cli.command
def install(
    bundle: t.Annotated[
        Path,
        cyclopts.Parameter(help="Path to Zarf package bundle"),
    ],
    ecr_registry: t.Annotated[
        str,
        cyclopts.Parameter(help="ECR registry URL (e.g., 123456.dkr.ecr.us-east-1.amazonaws.com)"),
    ],
    namespace: t.Annotated[
        str,
        cyclopts.Parameter(help="Kubernetes namespace for deployment"),
    ] = "dreadnode",
    region: t.Annotated[
        str | None,
        cyclopts.Parameter(help="AWS region (auto-detected if not provided)"),
    ] = None,
    *,
    skip_preflight: t.Annotated[
        bool,
        cyclopts.Parameter(help="Skip pre-flight validation checks"),
    ] = False,
    skip_health_check: t.Annotated[
        bool,
        cyclopts.Parameter(help="Skip post-install health verification"),
    ] = False,
) -> None:
    """
    Install Dreadnode platform from air-gapped bundle.

    This command deploys the platform to a Kubernetes cluster using the provided
    bundle and ECR registry for image storage.
    """
    if not bundle.exists():
        print_error(f"Bundle not found: {bundle}")
        raise SystemExit(1)

    try:
        installer = AirGapInstaller(
            bundle_path=bundle,
            ecr_registry=ecr_registry,
            namespace=namespace,
            region=region,
        )

        installer.install(
            skip_preflight=skip_preflight,
            skip_health_check=skip_health_check,
        )

    except ValidationError as e:
        # Validation errors are already formatted nicely
        raise SystemExit(1) from e
    except RuntimeError as e:
        print_error(f"Installation failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@cli.command
def health_check(
    namespace: t.Annotated[
        str,
        cyclopts.Parameter(help="Kubernetes namespace to check"),
    ] = "dreadnode",
    timeout: t.Annotated[
        int,
        cyclopts.Parameter(help="Timeout in seconds"),
    ] = 300,
) -> None:
    """
    Check health of deployed Dreadnode platform.

    Verifies that all pods are running and healthy.
    """
    from dreadnode.airgap.health import HealthChecker

    try:
        print_info(f"Checking platform health in namespace '{namespace}'...")

        health = HealthChecker(namespace)

        # Check pod status
        health.wait_for_ready(timeout=timeout)

        # Verify API
        health.verify_api_health()

        # Get summary
        summary = health.get_deployment_summary()

        console.print("\n[bold green]✅ Platform is healthy[/bold green]\n")
        console.print(f"  Pods: {summary['pods']['total']}")
        console.print(f"  Services: {summary['services']['total']}")
        console.print(f"  Deployments: {summary['deployments']['total']}")

        print_success("Health check passed")

    except Exception as e:
        print_error(f"Health check failed: {e}")
        raise SystemExit(1) from e


@cli.command
def validate(
    bundle: t.Annotated[
        Path | None,
        cyclopts.Parameter(help="Path to bundle to validate"),
    ] = None,
    ecr_registry: t.Annotated[
        str | None,
        cyclopts.Parameter(help="ECR registry URL to validate connectivity"),
    ] = None,
    *,
    skip_k8s: t.Annotated[
        bool,
        cyclopts.Parameter(help="Skip Kubernetes connectivity checks"),
    ] = False,
) -> None:
    """
    Run pre-flight validation checks.

    Validates environment is ready for air-gapped deployment.
    """
    try:
        print_info("Running pre-flight validation checks...")

        validator = PreFlightValidator()
        ecr_helper = None

        if ecr_registry:
            ecr_helper = ECRHelper(ecr_registry)

        validator.validate_all(
            ecr_helper=ecr_helper,
            bundle_path=bundle,
            skip_k8s=skip_k8s,
        )

        print_success("All validation checks passed")

    except ValidationError as e:
        # Validation errors are already formatted
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Validation failed: {e}")
        raise SystemExit(1) from e
