"""
Axolotl integration for Dreadnode.

This plugin enables logging training metrics and parameters to Dreadnode Strikes
when using Axolotl for fine-tuning. It injects the existing DreadnodeCallback
into Axolotl's trainer via the plugin system.

Usage in axolotl config:
    plugins:
      - dreadnode.integrations.axolotl.DreadnodePlugin

    dreadnode_project: "my-project"
    dreadnode_run_name: "training-run-v1"  # optional
    dreadnode_tags:  # optional
      - "experiment-1"
"""

from __future__ import annotations

import logging
import typing as t

LOG = logging.getLogger(__name__)


class DreadnodePlugin:
    """
    Axolotl plugin that integrates DreadnodeCallback for training metrics logging.

    This plugin follows the same pattern as Axolotl's built-in integrations
    (SwanLab, etc.) and reuses the existing DreadnodeCallback from the
    transformers integration.
    """

    def __init__(self) -> None:
        self._callback: t.Any = None
        self._initialized = False

    def register(self, cfg: dict) -> None:
        """Called during plugin registration with unparsed config dict."""
        if cfg.get("dreadnode_project"):
            LOG.info("Dreadnode plugin registered for project: %s", cfg.get("dreadnode_project"))

    def get_input_args(self) -> str:
        """Return the dotted path to args class for config validation."""
        return "dreadnode.integrations.axolotl.args.DreadnodeAxolotlArgs"

    def pre_model_load(self, cfg: t.Any) -> None:
        """Called before model loading - early validation and logging."""
        if not getattr(cfg, "dreadnode_project", None):
            LOG.debug("Dreadnode integration disabled (no dreadnode_project set)")
            return

        LOG.info(f"Dreadnode integration enabled for project: {cfg.dreadnode_project}")

    def add_callbacks_post_trainer(self, cfg: t.Any, trainer: t.Any) -> list[t.Any]:
        """
        Hook called after trainer is created - inject our callback here.

        Returns a list of TrainerCallbacks that Axolotl will add to the trainer.
        """
        if not getattr(cfg, "dreadnode_project", None):
            return []

        # Only initialize on rank 0 in distributed training
        try:
            from axolotl.utils.distributed import is_main_process

            if not is_main_process():
                LOG.debug("Skipping Dreadnode callback on non-main process")
                return []
        except ImportError:
            # Fallback for older axolotl versions or non-distributed
            pass

        from dreadnode.integrations.transformers import DreadnodeCallback

        self._callback = DreadnodeCallback(
            project=cfg.dreadnode_project,
            run_name=getattr(cfg, "dreadnode_run_name", None),
            tags=getattr(cfg, "dreadnode_tags", None),
        )
        self._initialized = True

        LOG.info(
            f"Registered DreadnodeCallback: project={cfg.dreadnode_project}, "
            f"run_name={getattr(cfg, 'dreadnode_run_name', 'auto')}"
        )

        return [self._callback]

    def post_train(self, cfg: t.Any, model: t.Any) -> None:
        """Called after training completes."""
        if self._initialized:
            LOG.info("Training complete, Dreadnode run finalized")


__all__ = ["DreadnodePlugin"]
