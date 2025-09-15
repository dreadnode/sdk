import inspect
import typing as t
from copy import deepcopy

import typing_extensions as te

from dreadnode.meta import Component, ConfigInfo, Context
from dreadnode.util import warn_at_user_stacklevel

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)
In_contra = te.TypeVar("In_contra", contravariant=True, default=t.Any)
Out_co = te.TypeVar("Out_co", covariant=True, default=t.Any)
OuterIn = te.TypeVar("OuterIn", default=t.Any)
OuterOut = te.TypeVar("OuterOut", default=t.Any)


class TransformWarning(UserWarning):
    """Warning issued for non-critical issues during transformations."""


@t.runtime_checkable
class TransformCallable(t.Protocol, t.Generic[In_contra, Out_co]):
    """
    A callable that takes in one object, and returns another.
    """

    def __call__(
        self, obj: In_contra, /, *args: t.Any, **kwargs: t.Any
    ) -> t.Awaitable[Out_co] | Out_co: ...


class Transform(Component[te.Concatenate[In, ...], Out], t.Generic[In, Out]):
    """
    Represents a transformation operation that modifies the input data.
    """

    def __init__(
        self,
        func: TransformCallable[In, Out],
        *,
        name: str | None = None,
        catch: bool = False,
        config: dict[str, ConfigInfo] | None = None,
        context: dict[str, Context] | None = None,
    ):
        super().__init__(
            t.cast("t.Callable[[In], Out]", func), name=name, config=config, context=context
        )

        self.name = self.name
        "The name of the transform, used for reporting and logging."
        self.catch = catch
        """
        If True, catches exceptions during the transform and attempts to return the original,
        unmodified object from the input. If False, exceptions are raised.
        """

    @classmethod
    def fit(cls, transform: "TransformLike[In, Out]") -> "Transform[In, Out]":
        """Ensures that the provided transform is a Transform instance."""
        if isinstance(transform, Transform):
            return transform
        if callable(transform):
            return Transform(transform)
        raise TypeError("Transform must be a Transform instance or a callable.")

    @classmethod
    def fit_many(cls, transforms: "TransformsLike[In, Out] | None") -> list["Transform[In, Out]"]:
        """
        Convert a collection of transform-like objects into a list of Transform instances.

        This method provides a flexible way to handle different input formats for transforms,
        automatically converting callables to Transform objects and applying consistent naming
        and attributes across all transforms.

        Args:
            transforms: A collection of transform-like objects. Can be:
                - A dictionary mapping names to transform objects or callables
                - A sequence of scorer objects or callables
                - None (returns empty list)

        Returns:
            A list of Scorer instances with consistent configuration.
        """
        if isinstance(transforms, t.Mapping):
            return [
                transform.with_(name=name)
                if isinstance(transform, Transform)
                else cls(transform, name=name)
                for name, transform in transforms.items()
            ]

        return [
            transform if isinstance(transform, Transform) else cls(transform)
            for transform in transforms or []
        ]

    def __repr__(self) -> str:
        return f"Transform(name='{self.name}')"

    def __deepcopy__(self, memo: dict[int, t.Any]) -> "Transform[In, Out]":
        return Transform(
            func=self.func,
            name=self.name,
            catch=self.catch,
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> "Transform[In, Out]":
        """Clone the transform."""
        return self.__deepcopy__({})

    def with_(
        self,
        *,
        name: str | None = None,
        catch: bool | None = None,
    ) -> "Transform[In, Out]":
        """
        Create a new Transform with updated properties.

        Args:
            name: New name for the transform.
            catch: Catch exceptions in the transform function.

        Returns:
            A new Transform with the updated properties
        """
        new = self.clone()
        new.name = name or self.name
        new.catch = catch or self.catch
        return new

    def rename(self, new_name: str) -> "Transform[In, Out]":
        """
        Rename the transform.

        Args:
            new_name: The new name for the transform.

        Returns:
            A new Transform with the updated name.
        """
        return self.with_(name=new_name)

    def adapt(
        self: "Transform[In, Out]",
        adapt_in: t.Callable[[OuterIn], In],
        adapt_out: t.Callable[[Out], OuterOut],
        name: str | None = None,
    ) -> "Transform[OuterIn, OuterOut]":
        """
        Adapts a transform to operate with some other in/out types.

        This is a powerful wrapper that allows a generic transform (e.g., one that
        refines a string) to be used with a complex candidate object (e.g., a
        Pydantic model containing that string).

        Args:
            adapt_in: A function to extract the `T` from the `OuterT`.
            adapt_out: A function to extract the `OuterT` from the `T`.
            name: An optional new name for the adapted scorer.

        Returns:
            A new Scorer instance that operates on the `OuterT`.
        """
        original = self

        async def transform(object: OuterIn, *args: t.Any, **kwargs: t.Any) -> OuterOut:
            adapted = adapt_in(object)
            result = await original.transform(adapted, *args, **kwargs)
            return adapt_out(result)

        return Transform(transform, name=name or self.name)

    async def transform(self, object: In, *args: t.Any, **kwargs: t.Any) -> Out:
        """
        Perform a transform from In to Out.

        Args:
            object: The input object to transform.

        Returns:
            The transformed output object.
        """
        # TODO(nick): Should consider creating a task or span here to
        # track this operation

        try:
            bound_args = self._bind_args(object, *args, **kwargs)
            result = t.cast(
                "Out | t.Awaitable[Out]", self.func(*bound_args.args, **bound_args.kwargs)
            )
            if inspect.isawaitable(result):
                result = await result

        except Exception as e:
            if not self.catch:
                raise

            # As a fallback, attempt to return the original object
            #
            # TODO(nick): Is this behavior ideal? We could do some generic
            # inspection to see if this is reasonable, but it might make
            # more sense to just error here

            warn_at_user_stacklevel(
                f"Error executing transformation {self.name!r} for object {object!r}: {e}",
                TransformWarning,
            )
            return t.cast("Out", object)

        return result

    @te.override
    async def __call__(self, object: In, *args: t.Any, **kwargs: t.Any) -> Out:  # type: ignore[override]
        return await self.transform(object, *args, **kwargs)

    # TODO(nick): Implement composition mechanics here like chaining or random selection


TransformLike = Transform[In, Out] | TransformCallable[In, Out]
"""A transform or compatible callable."""
TransformsLike = t.Sequence[TransformLike[In, Out]] | t.Mapping[str, TransformLike[In, Out]]
"""A sequence of transform-like objects or mapping of name/transform pairs."""
