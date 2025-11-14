import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from oras.client import OrasClient

from dreadnode.api.models import ContainerRegistryCredentials
from dreadnode.constants import DATASETS_CACHE
from dreadnode.dataset import Dataset

# --- Setup Logging and Custom Exceptions ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatasetManagerError(Exception):
    """Base exception for our DatasetManager."""


class AuthenticationError(DatasetManagerError):
    """Raised for authentication failures."""


class DatasetNotFoundError(DatasetManagerError):
    """Raised when a dataset is not found in the cache or registry."""


# --- Advanced Authentication ---


class OciDatasetStorage:
    """
    Manages the lifecycle of Parquet datasets as OCI artifacts.

    This class handles pushing and pulling datasets from local and in-memory
    sources, while enriching the artifacts with schema and dependency metadata.
    """

    def __init__(self, org, credential_fetcher: Callable[[], "ContainerRegistryCredentials"]):
        self.credential_fetcher = credential_fetcher
        self.creds = None
        self.registry = None
        self.cache_root = DATASETS_CACHE
        self.org = org

    def _login(self):
        try:
            self.creds = self.credential_fetcher()
            self.client = OrasClient()
            self.registry = self.creds.registry
            self.client.login(
                hostname=self.creds.registry,
                username=self.creds.username,
                password=self.creds.password,
            )
            logging.info(f"Authenticated to OCI registry: {self.registry}")
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize ORAS client: {e}")

    def _get_cache_path(self, dataset_name: str, version: str) -> Path:
        """Constructs the canonical path for a dataset in the local cache."""
        return self.cache_root / dataset_name / version

    def push(
        self,
        dataset: Dataset,
        version: str = "latest",
        custom_meta: dict | None = None,
    ) -> str:
        """
        Prepares, packages, and pushes a dataset to the OCI registry.

        This single method elegantly handles all three of your specified scenarios.

        Args:
            dataset_name: Name of the dataset.
            version: Version tag for the dataset.
            source: The dataset source. Can be:
                - A path to a folder (Scenario 1).
                - A pandas DataFrame (Scenario 3, single file).
                - A dict of {filename: DataFrame} (Scenario 3, multiple files).
            dependencies: A list of dependent dataset URIs (e.g., ['<registry>/<org>/<name>:<tag>']).
            custom_meta: A dictionary for any other metadata you want to include.

        Returns:
            The manifest digest of the pushed artifact.
        """

        files = dataset.save(f"{DATASETS_CACHE}/{self.org}/{dataset.name}/{version}")
        # os.chdir(f"{DATASETS_CACHE}/{self.org}/{dataset.name}/{version}")

        # --- Step 3: Push using ORAS Client ---

        import os

        os.chdir(f"{DATASETS_CACHE}/{self.org}/{dataset.name}/{version}")
        self._login()
        target_ref = f"{self.registry}/datasets/{self.org}/{dataset.name}:{version}"
        logging.info(f"Pushing dataset to {target_ref}...")
        manifest_digest = self.client.push(
            files=files,
            target=target_ref,
        )
        logging.info(f"Successfully pushed. Manifest digest:\n {manifest_digest.content}")
        return manifest_digest

    def pull(self, dataset_name: str, version: str, force: bool = False) -> dict:
        """
        Pulls a dataset from the OCI registry into the local cache.

        Args:
            dataset_name: The name of the dataset.
            version: The version tag of the dataset.
            force: If True, overwrite existing files in the cache.

        Returns:
            A dictionary containing the local path and retrieved metadata.
        """
        source_ref = f"{self.registry}/{self.org}/{dataset_name}:{version}"
        cache_path = self._get_cache_path(dataset_name, version)

        if cache_path.exists() and not force:
            logging.info(
                f"Dataset '{dataset_name}:{version}' already exists in cache. Skipping pull."
            )
        else:
            cache_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Pulling dataset from {source_ref} to {cache_path}...")
            try:
                self.client.pull(target=source_ref, outdir=str(cache_path))
                logging.info("Pull successful.")
            except Exception as e:
                raise DatasetManagerError(f"Failed to pull dataset: {e}")

        # --- Always fetch metadata to return it ---
        metadata = self.get_metadata(dataset_name, version)
        return {
            "path": str(cache_path),
            "metadata": metadata,
        }

    def get_metadata(self, dataset_name: str, version: str) -> dict:
        """
        Fetches only the metadata (manifest and config) for a remote dataset.
        This is highly efficient as it doesn't download the file layers.
        """
        source_ref = f"{self.registry}/{self.org}/{dataset_name}:{version}"
        try:
            manifest = self.client.manifest_get(target=source_ref)
            config_bytes = self.client.blob_get(
                target=source_ref, digest=manifest["config"]["digest"]
            )
            return {"manifest": manifest, "config": json.loads(config_bytes)}
        except FileNotFoundError:
            raise DatasetNotFoundError(f"Dataset not found in registry: {source_ref}")


class TempStaging:
    """A context manager to safely stage files for pushing."""

    def __init__(self, source, final_cache_path):
        self.source = source
        self.final_cache_path = final_cache_path
        self.staging_path = Path(str(final_cache_path) + "_staging")

    def __enter__(self):
        if self.staging_path.exists():
            shutil.rmtree(self.staging_path)
        self.staging_path.mkdir(parents=True)

        if isinstance(self.source, (str, Path)):  # Scenario 1: External Folder
            source_path = Path(self.source).expanduser()
            if not source_path.is_dir():
                raise ValueError(f"Source path '{source_path}' is not a directory.")
            shutil.copytree(source_path, self.staging_path, dirs_exist_ok=True)
        elif isinstance(self.source, pd.DataFrame):  # Scenario 3: In-memory DataFrame
            self.source.to_parquet(self.staging_path / "data.parquet")
        elif isinstance(self.source, dict):  # Scenario 3: Multiple in-memory DataFrames
            for filename, df in self.source.items():
                df.to_parquet(self.staging_path / filename)
        else:
            raise TypeError(f"Unsupported source type: {type(self.source)}")
        return self.staging_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # If no exception, commit staging to final cache
            if self.final_cache_path.exists():
                shutil.rmtree(self.final_cache_path)
            shutil.move(str(self.staging_path), self.final_cache_path)
        else:  # On error, clean up staging
            shutil.rmtree(self.staging_path)
