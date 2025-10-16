import typing as t

from loguru import logger

from dreadnode.meta import Config
from dreadnode.optimization.collectors import lineage, local_neighborhood
from dreadnode.optimization.sampling import interleave_by_parent, top_k
from dreadnode.optimization.search.base import OptimizationContext, Search
from dreadnode.optimization.trial import CandidateT, Trial, TrialCollector, TrialSampler
from dreadnode.transforms import Transform, TransformLike
from dreadnode.util import concurrent_gen


def graph_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    branching_factor: int = 3,
    context_collector: TrialCollector[CandidateT] = lineage,
    pruning_sampler: TrialSampler[CandidateT] = top_k,
    name: str = "graph",
) -> Search[CandidateT]:
    """
    Creates a generalized, stateful strategy for generative graph-based search.

    Formally, the structure is a connected directed acyclic graph (DAG) where nodes represent
    trials and edges are parent-child relationships.

    For each iteration, it:
        1 - Gathers related trials using `context_collector` for every leaf node
        2 - Applies the `transform` to [leaf, *context] `branching_factor` times for each leaf
        3 - Suggests all new children for evaluation
        4 - Waits for all children to complete
        5 - Prunes with `pruning_sampler` to establish leaves for the next step
    """

    async def search(
        _: OptimizationContext,
        *,
        transform: TransformLike[list[Trial[CandidateT]], CandidateT] = Config(transform),  # noqa: B008
        initial_candidate: CandidateT = Config(initial_candidate),  # noqa: B008
        branching_factor: int = Config(branching_factor),
        context_collector: TrialCollector[CandidateT] = Config(context_collector),  # noqa: B008
        pruning_sampler: TrialSampler[CandidateT] = Config(pruning_sampler),  # noqa: B008
    ) -> t.AsyncGenerator[Trial[CandidateT], None]:
        trials: list[Trial[CandidateT]] = []
        leaves: list[Trial[CandidateT]] = []
        transform = Transform.fit(transform)

        logger.info(
            "Starting graph search: "
            f"branching_factor={branching_factor}, "
            f"context_collector={context_collector}, "
            f"pruning_sampler={pruning_sampler}"
        )

        initial_trial = Trial(candidate=initial_candidate)
        yield initial_trial
        await initial_trial

        if initial_trial.status != "finished":
            return

        trials.append(initial_trial)
        leaves = [initial_trial]

        while leaves:
            # Generate all new trials branching from current leaves
            new_trials: list[Trial[CandidateT]] = []
            for leaf in leaves:
                trials_context = [leaf, *context_collector(leaf, trials)]
                coroutines = [transform(trials_context) for _ in range(branching_factor)]
                async with concurrent_gen(coroutines) as gen:
                    async for candidate in gen:
                        new_trial = Trial(candidate=candidate, parent_id=leaf.id)
                        new_trials.append(new_trial)
                        yield new_trial

            # Wait for all new trials to complete
            await Trial.wait_for(*new_trials)

            # Collect finished trials and prune to get new leaves
            finished = [t for t in new_trials if t.status == "finished"]
            trials.extend(finished)
            interleaved = interleave_by_parent(finished)
            leaves = pruning_sampler(interleaved)

    return Search(search, name=name)


def iterative_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    branching_factor: int = 1,
    parent_depth: int = 10,
) -> Search[CandidateT]:
    """
    Creates a GraphSearch configured for single-path iterative refinement.

    This strategy maintains a single path of improvement by always expanding from the
    single best trial of the previous step. The context for refinement is the
    direct lineage of that best trial.

    Set `branching_factor` > 1 to explore multiple candidates at each step.

    Args:
        transform: The function that takes the history and generates new candidates.
        initial_candidate: The starting point for the refinement chain.
        branching_factor: How many new candidates to generate from the best trial at each step.
                          The best of these will be chosen for the next step.
        parent_depth: The number of direct ancestors to include in the context for refinement.

    Returns:
        A pre-configured graph search instance.
    """
    return graph_search(
        transform=transform,
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=lineage.configure(depth=parent_depth),
        pruning_sampler=top_k.configure(k=1),
        name="iterative",
    )


def beam_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    beam_width: int = 3,
    branching_factor: int = 3,
    parent_depth: int = 10,
) -> Search[CandidateT]:
    """
    Creates a graph search configured for classic beam search.

    This strategy maintains parallel reasoning paths by keeping a "beam" of the top `k`
    best trials from the previous step. Each trial in the beam is expanded independently,
    using its own lineage for context.

    Args:
        transform: The function that takes the history and generates new candidates.
        initial_candidate: The starting point for the refinement chain.
        beam_width: The number of top candidates to keep at each step (the 'k').
        branching_factor: How many new candidates to generate from each trial in the beam.
        parent_depth: The number of direct ancestors to include in the context for refinement.

    Returns:
        A pre-configured GraphSearch instance.
    """
    return graph_search(
        transform=transform,
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=lineage.configure(depth=parent_depth),
        pruning_sampler=top_k.configure(k=beam_width),
        name="beam",
    )


def graph_neighborhood_search(
    transform: TransformLike[list[Trial[CandidateT]], CandidateT],
    initial_candidate: CandidateT,
    *,
    neighborhood_depth: int = 2,
    frontier_size: int = 5,
    branching_factor: int = 3,
) -> Search[CandidateT]:
    """
    Creates a graph search configured with a local neighborhood context, where the trial context
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
    return graph_search(
        transform=transform,
        initial_candidate=initial_candidate,
        branching_factor=branching_factor,
        context_collector=local_neighborhood.configure(depth=neighborhood_depth),
        pruning_sampler=top_k.configure(k=frontier_size),
        name="graph_neighborhood",
    )
