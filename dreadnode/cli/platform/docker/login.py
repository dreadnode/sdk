import subprocess
import sys

import rich

from dreadnode.cli.api import create_api_client


def docker_login():
    client = create_api_client()
    container_registry_creds = client.get_container_registry_credentials()

    cmd = ["docker", "login", container_registry_creds.registry]
    cmd.extend(["--username", container_registry_creds.username])
    cmd.extend(["--password-stdin"])

    try:
        subprocess.run(cmd, input=container_registry_creds.password, text=True, check=True)  # noqa: S603
        rich.print(f"Logged in to Docker registry: {container_registry_creds.registry}")
    except subprocess.CalledProcessError as e:
        rich.print(f"Failed to log in to Docker registry: {e}", file=sys.stderr)
        raise
