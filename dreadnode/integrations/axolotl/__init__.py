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

    def get_training_args_mixin(self) -> None:
        """Returns training args mixin class - not used by Dreadnode plugin."""
        return None

    def load_datasets(self, cfg: t.Any, preprocess: bool = False) -> None:
        """Not used by Dreadnode plugin - returns None to use default loading."""
        return None

    def pre_model_load(self, cfg: t.Any) -> None:
        """Called before model loading - early validation and logging."""
        if not getattr(cfg, "dreadnode_project", None):
            LOG.debug("Dreadnode integration disabled (no dreadnode_project set)")
            return

        LOG.info(f"Dreadnode integration enabled for project: {cfg.dreadnode_project}")

    def post_model_build(self, cfg: t.Any, model: t.Any) -> None:
        """Called after model is built - not used by Dreadnode plugin."""
        pass

    def pre_lora_load(self, cfg: t.Any, model: t.Any) -> None:
        """Called before LoRA loading - not used by Dreadnode plugin."""
        pass

    def post_lora_load(self, cfg: t.Any, model: t.Any) -> None:
        """Called after LoRA loading - not used by Dreadnode plugin."""
        pass

    def post_model_load(self, cfg: t.Any, model: t.Any) -> None:
        """Called after model loading - not used by Dreadnode plugin."""
        pass

    def get_trainer_cls(self, cfg: t.Any) -> None:
        """Returns custom trainer class - not used by Dreadnode plugin."""
        return None

    def post_trainer_create(self, cfg: t.Any, trainer: t.Any) -> None:
        """Called after trainer creation - not used by Dreadnode plugin."""
        pass

    def get_training_args(self, cfg: t.Any) -> None:
        """Returns custom training args - not used by Dreadnode plugin."""
        return None

    def get_collator_cls_and_kwargs(self, cfg: t.Any, is_eval: bool = False) -> None:
        """Returns custom collator - not used by Dreadnode plugin."""
        return None

    def create_optimizer(self, cfg: t.Any, trainer: t.Any) -> None:
        """Returns custom optimizer - not used by Dreadnode plugin."""
        return None

    def create_lr_scheduler(
        self, cfg: t.Any, trainer: t.Any, optimizer: t.Any, num_training_steps: int
    ) -> None:
        """Returns custom LR scheduler - not used by Dreadnode plugin."""
        return None

    def add_callbacks_pre_trainer(self, cfg: t.Any, model: t.Any) -> list[t.Any]:
        """Returns callbacks before trainer creation - not used by Dreadnode plugin."""
        return []

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

        import dreadnode as dn
        from dreadnode.integrations.transformers import DreadnodeCallback

        # Configure workspace if provided
        workspace = getattr(cfg, "dreadnode_workspace", None)
        if workspace:
            dn.configure(workspace=workspace)

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

    def post_train_unload(self, cfg: t.Any) -> None:
        """Called after training unload - not used by Dreadnode plugin."""
        pass


__all__ = ["DreadnodePlugin"]
