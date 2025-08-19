import inspect
import typing as t
from dataclasses import dataclass

from logfire._internal.stack_info import warn_at_user_stacklevel
from logfire._internal.utils import safe_repr

# Import from your framework's core modules
from dreadnode.configurable import clone_config_attrs
from dreadnode.optimization import Trial

# Define generic type for the candidate's state
CandidateT = t.TypeVar("CandidateT")


class MutationWarning(UserWarning):
    """Warning issued for non-critical issues during mutation."""


# --- Core Type Definitions ---

MutationCallable = t.Callable[["Trial[CandidateT]"], t.Awaitable[CandidateT]]
"""
A callable that takes a completed Trial (containing the candidate state, score, etc.)
and returns a new, mutated candidate state.
"""

MutationLike = t.Union["Mutation[CandidateT]", MutationCallable[CandidateT]]
"""A type hint for anything that can be resolved into a Mutation."""


@dataclass
class Mutation(t.Generic[CandidateT]):
    """
    Represents a generative, novelty-producing operation.

    Mutations are stateful callables that take a Trial as input and produce a new
    candidate state, forming the core of generative search algorithms.
    """

    name: str
    """The name of the mutation, used for identification and logging."""
    func: MutationCallable[CandidateT]
    """The async function to call to perform the mutation."""
    catch: bool = False
    """
    If True, catches exceptions during the mutation and returns the original,
    unmodified candidate from the input trial. If False, exceptions are raised.
    """

    @classmethod
    def from_callable(
        cls,
        func: "MutationLike[CandidateT]",
        *,
        name: str | None = None,
        catch: bool = False,
    ) -> "Mutation[CandidateT]":
        """
        Create a Mutation from a callable function.

        This is the primary factory for turning a simple async function into a
        full-featured Mutation object.

        Args:
            func: The async function that performs the mutation logic.
            name: The name of the mutation. If not provided, it's inferred from the function name.
            catch: Whether to catch exceptions during mutation.

        Returns:
            A Mutation object.
        """
        if isinstance(func, Mutation):
            return func

        unwrapped = inspect.unwrap(func)
        func_name = getattr(
            unwrapped, "__qualname__", getattr(func, "__name__", safe_repr(unwrapped))
        )

        name = name or func_name
        return clone_config_attrs(
            func,
            cls(
                name=name,
                func=func,
                catch=catch,
            ),
        )

    def __post_init__(self) -> None:
        """Ensures the instance has introspection-friendly attributes."""
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Mutation[CandidateT]":
        """
        Create an exact copy of this Mutation.

        Returns:
            A new Mutation instance with the same configuration.
        """
        return clone_config_attrs(
            self,
            Mutation(
                name=self.name,
                func=self.func,
                catch=self.catch,
            ),
        )

    def with_(
        self,
        name: str | None = None,
        catch: bool | None = None,
    ) -> "Mutation[CandidateT]":
        """
        Create a new Mutation with updated properties.

        Args:
            name: New name for the mutation.
            catch: Override the exception catching behavior.

        Returns:
            A new Mutation with the updated properties.
        """
        new = self.clone()
        new.name = name or self.name
        new.func = self.func
        new.catch = catch if catch is not None else self.catch
        return new

    def rename(self, new_name: str) -> "Mutation[CandidateT]":
        """
        Create a new Mutation with a different name.

        Args:
            new_name: The new name for the mutation.

        Returns:
            A new Mutation with the updated name.
        """
        return self.with_(name=new_name)

    async def mutate(self, trial: "Trial[CandidateT]") -> CandidateT:
        """
        Executes the mutation logic on a given trial.

        This is the core method of the Mutation. It takes the full context of a
        previous trial (candidate, score, status) and generates the next candidate.

        Args:
            trial: The completed Trial object to mutate from.

        Returns:
            A new candidate state of type CandidateT.
        """
        try:
            result = self.func(trial)
            if inspect.isawaitable(result):
                return await result
            raise TypeError(f"Mutation function for '{self.name}' must be async.")
        except Exception as e:
            if not self.catch:
                raise

            warn_at_user_stacklevel(
                f"Error executing mutation {self.name!r} for trial candidate {trial.candidate_state!r}: {e}",
                MutationWarning,
            )
            # As a safe fallback, return the original, unmodified candidate.
            return trial.candidate_state

    async def __call__(self, trial: Trial[CandidateT]) -> CandidateT:
        """
        Allows the Mutation instance to be called directly like a function.

        Args:
            trial: The completed Trial object to mutate from.

        Returns:
            A new candidate state of type CandidateT.
        """
        return await self.mutate(trial)
