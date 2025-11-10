from pathlib import Path

import cyclopts

cli = cyclopts.App("dataset", help="Run and manage datasets.")


@cli.command(name="list")
def list() -> None:
    """
    List available datasets on the Dreadnode platform.
    """
    print("Listing datasets available on the Dreadnode platform.")


@cli.command(name="push")
def push(dataset: Path) -> None:
    """
    Push a dataset to the Dreadnode platform.
    """
    print(f"Pushing dataset from {dataset} to Dreadnode platform.")


@cli.command(name="pull")
def pull(dataset_id: str, destination: Path | None = None) -> None:
    """
    Pull a dataset from the Dreadnode platform.
    """

    # def log_artifact(
    #     self,
    #     local_uri: str | Path,
    # ) -> None:
    # """
    #     Logs a local file or directory as an artifact to the object store.
    #     Preserves directory structure and uses content hashing for deduplication.

    #     Args:
    #         local_uri: Path to the local file or directory

    #     Returns:
    #         DirectoryNode representing the artifact's tree structure

    #     Raises:
    #         FileNotFoundError: If the path doesn't exist
    #     """
    #     artifact_tree = self._artifact_tree_builder.process_artifact(local_uri)
    #     self._artifact_merger.add_tree(artifact_tree)
    #     self._artifacts = self._artifact_merger.get_merged_trees()
