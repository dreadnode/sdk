import typing as t

from dreadnode.eval.events import SamplePostProcess, SamplePreProcess
from dreadnode.eval.reactions import ModifyInput, ModifyOutput
from dreadnode.transforms import Transform
from dreadnode.transforms.multimodal import apply_transforms_to_kwargs, apply_transforms_to_value

if t.TYPE_CHECKING:
    from dreadnode.eval.events import EvalEvent
    from dreadnode.eval.hooks.base import EvalHook
    from dreadnode.eval.reactions import EvalReaction


def apply_transforms(
    transforms: list[Transform],
    *,
    stage: t.Literal["input", "output"],
    create_task: bool = True,
) -> "EvalHook":
    """
    Creates a hook that applies transforms at the specified stage.
    """

    async def hook(event: "EvalEvent") -> "EvalReaction | None":  # noqa: PLR0911
        """Hook implementation that applies transforms based on stage."""

        # Input stage
        if stage == "input":
            if not isinstance(event, SamplePreProcess):
                return None

            if not transforms:
                return None

            if create_task:
                from dreadnode import task as dn_task

                task_kwargs = event.task_kwargs

                @dn_task(
                    name=f"transform - input ({len(transforms)} transforms)",
                    tags=["transform", "input", "hook"],
                    log_inputs=True,
                    log_output=True,
                )
                async def apply_task(
                    data: dict[str, t.Any] = task_kwargs,  # Use extracted variable
                ) -> dict[str, t.Any]:
                    return await apply_transforms_to_kwargs(data, transforms)

                transformed = await apply_task()
                return ModifyInput(task_kwargs=transformed)

            # Direct application
            transformed = await apply_transforms_to_kwargs(event.task_kwargs, transforms)
            return ModifyInput(task_kwargs=transformed)

        # Output stage
        if not isinstance(event, SamplePostProcess):
            return None

        if not transforms or event.output is None:
            return None

        if create_task:
            from dreadnode import task as dn_task

            output_data = event.output  # Extract before task decorator

            @dn_task(
                name=f"transform - output ({len(transforms)} transforms)",
                tags=["transform", "output", "hook"],
                log_inputs=True,
                log_output=True,
            )
            async def apply_task(data: t.Any = output_data) -> t.Any:  # Use extracted variable
                return await apply_transforms_to_value(data, transforms)

            transformed = await apply_task()
            return ModifyOutput(output=transformed)

        # Direct application
        transformed = await apply_transforms_to_value(event.output, transforms)
        return ModifyOutput(output=transformed)

    return hook


def apply_input_transforms(
    transforms: list[Transform],
    *,
    create_task: bool = True,
) -> "EvalHook":
    """Convenience function for input transforms."""
    return apply_transforms(transforms, stage="input", create_task=create_task)


def apply_output_transforms(
    transforms: list[Transform],
    *,
    create_task: bool = True,
) -> "EvalHook":
    """Convenience function for output transforms."""
    return apply_transforms(transforms, stage="output", create_task=create_task)
