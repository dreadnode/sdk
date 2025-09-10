import typing as t

from pydantic import ConfigDict, PrivateAttr

from dreadnode.meta import Config, Model
from dreadnode.optimization.collectors import lineage, local_neighborhood
from dreadnode.optimization.sampling import interleave_by_parent, top_k
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import CandidateT, Trial, TrialCollector, TrialSampler
from dreadnode.transforms import Transform, TransformLike
from dreadnode.util import concurrent_gen, get_callable_name


class GraphSearch(Model, Search[CandidateT]):
    """
    A generalized, stateful strategy for generative graph-based search.

    Formally, the structure is a connected directed acyclic graph (DAG) where nodes represent
    trials and edges are parent-child relationships.

    For each step, it:
        1 - Gathers related trials using `context_collector` for every leaf node
        2 - Applies the `transform` to [leaf, *context] `branching_factor` times for each leaf
        3 - Suggests all new children for evaluation

    When trials are observed, it:
        1 - Filters out non-completed trials
        2 - Adds new children to the graph
        3 - Prunes with `pruning_sampler` to establish leaves for the next step
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    transform: Transform[list[Trial[CandidateT]], CandidateT]
    """The transform for generating new nodes from the current trial and related context."""
    initial_candidate: CandidateT
    """The initial candidate for the search."""

    branching_factor: int = Config(default=3)
    """The number of new candidates to generate from each leaf node."""
    max_leaves: int = Config(default=10)
    """The maximum number of leaf nodes to maintain in the search."""

    context_collector: TrialCollector[CandidateT] = Config(lineage)
    """A trial collector to gather relevant trials before branching."""
    pruning_sampler: TrialSampler[CandidateT] = Config(top_k)
    """A trial sampler to prune new children after each branching."""

    _trials: list[Trial[CandidateT]] = PrivateAttr(default_factory=list)
    _leaves: list[Trial[CandidateT]] = PrivateAttr(default_factory=list)

    def __repr__(self) -> str:
        parts = [
            f"transform={get_callable_name(self.transform, short=True)}"
            f"context_collector={get_callable_name(self.context_collector, short=True)}"
            f"pruning_sampler={get_callable_name(self.pruning_sampler, short=True)}"
            f"branching_factor={self.branching_factor}"
        ]
        return f"GraphSearch({', '.join(parts)})"

    def reset(self, _: OptimizationContext) -> None:
        self._trials = []
        self._leaves = []

    async def suggest(self, step: int) -> t.AsyncIterator[Trial[CandidateT]]:
        if not self._leaves:
            yield Trial(candidate=self.initial_candidate, step=step)
            return

        for leaf in self._leaves:
            context = [leaf, *self.context_collector(leaf, self._trials)]
            coroutines = [self.transform(context) for _ in range(self.branching_factor)]
            async with concurrent_gen(coroutines) as gen:
                async for candidate in gen:
                    yield Trial(candidate=candidate, parent_id=leaf.id, step=step)

        return

    async def observe(self, trials: list[Trial[CandidateT]]) -> None:
        finished_trials = [t for t in trials if t.status == "finished"]
        self._trials.extend(finished_trials)
        interleaved_trials = interleave_by_parent(finished_trials)  # Prevent parent bias
        self._leaves = self.pruning_sampler(interleaved_trials)


def iterative_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    branching_factor: int = 1,
) -> GraphSearch[CandidateT]:
    """
    Creates a GraphSearch configured for single-path iterative refinement.

    This strategy maintains a single path of reasoning by always expanding from the
    single best trial of the previous step. The context for refinement is the
    direct lineage of that best trial.

    Set `branching_factor` > 1 to explore multiple candidates at each step.

    Args:
        transform: The function that takes the history and generates new candidates.
        initial_candidate: The starting point for the refinement chain.
        branching_factor: How many new candidates to generate from the best trial at each step.
                          The best of these will be chosen for the next step.

    Returns:
        A pre-configured GraphSearch instance.
    """
    return GraphSearch[CandidateT](
        transform=Transform.fit(transform),
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=lineage,
        pruning_sampler=top_k.configure(k=1),
    )


def beam_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    beam_width: int = 3,
    branching_factor: int = 3,
) -> GraphSearch[CandidateT]:
    """
    Creates a GraphSearch configured for classic beam search.

    This strategy maintains parallel reasoning paths by keeping a "beam" of the top `k`
    best trials from the previous step. Each trial in the beam is expanded independently,
    using its own lineage for context.

    Args:
        transform: The function that takes the history and generates new candidates.
        initial_candidate: The starting point for the refinement chain.
        beam_width: The number of top candidates to keep at each step (the 'k').
        branching_factor: How many new candidates to generate from each trial in the beam.

    Returns:
        A pre-configured GraphSearch instance.
    """
    return GraphSearch[CandidateT](
        transform=Transform.fit(transform),
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=lineage,
        pruning_sampler=top_k.configure(k=beam_width),
    )


def graph_neighborhood_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    neighborhood_depth: int = 2,
    frontier_size: int = 5,
    branching_factor: int = 3,
) -> GraphSearch[CandidateT]:
    """
    Creates a GraphSearch configured with a local neighborhood context, where the trial context
    passed to the transform includes the trials in the local neighborhood up to `2h-1` distance
    away where `h` is the neighborhood depth. This means the trials which are "parents",
    "grandparents", "uncles", or "cousins" can be considered during the creation of new nodes.

    Once the pool of candidate trials is established, `frontier_size` determines how many of
    the best candidates are kept for the iteration.

    See: "Graph of Attacks" - https://arxiv.org/pdf/2504.19019v1

    Args:
        transform: The function that takes the neighborhood context and generates new candidates.
        initial_candidate: The starting point for the search.
        neighborhood_depth: The depth 'h' used to calculate the size of the local neighborhood context.
        frontier_size: The number of top candidates to form the next generation's frontier ('d').
        branching_factor: How many new candidates to generate from each current leaf node.

    Returns:
        A pre-configured GraphSearch instance.
    """
    return GraphSearch[CandidateT](
        transform=Transform.fit(transform),
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=local_neighborhood.configure(depth=neighborhood_depth),
        pruning_sampler=top_k.configure(k=frontier_size),
    )
