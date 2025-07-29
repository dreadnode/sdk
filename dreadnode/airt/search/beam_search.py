import asyncio
import typing as t

import rigging as rg

from dreadnode.airt.attack import AttackConfig
from dreadnode.airt.search.base import Search


def beam_search(beam_width: int = 5, max_steps: int = 3) -> Search:
    """Perform a beam search attack on the model."""

    async def search(config: AttackConfig) -> float:
        highest_score_overall = -1.0

        for prompt in config.prompts:
            print(f"\n--- Attacking Prompt: '{prompt}' ---")

            initial_pipeline = config.build_pipeline(prompt)
            beams = [initial_pipeline.clone() for _ in range(beam_width)]
            best_chat_for_prompt: rg.Chat | None = None
            best_score_for_prompt = -1.0

            for step in range(max_steps):
                candidate_pipelines: list[rg.ChatPipeline] = []
                for beam in beams:
                    original_text = beam.chat.last.content
                    for _ in range(beam_width):
                        transformed_text = config.transforms[0](original_text)
                        if all(c(transformed_text, original_text) for c in config.constraints):
                            candidate_pipelines.append(beam.fork(transformed_text))

                if not candidate_pipelines:
                    break

                tasks = [candidate.run(on_failed="include") for candidate in candidate_pipelines]
                results = await asyncio.gather(*tasks)

                scored_results = []
                for chat in results:
                    if chat.failed:
                        continue
                    score = config.scorers[0](chat)
                    scored_results.append((score, chat))

                if not scored_results:
                    break

                scored_results.sort(key=lambda x: x[0], reverse=True)

                current_best_score, current_best_chat = scored_results[0]
                if current_best_score > best_score_for_prompt:
                    best_score_for_prompt = current_best_score
                    best_chat_for_prompt = current_best_chat

                print(f"  Step {step + 1}: Best score for prompt = {best_score_for_prompt:.2f}")

                if best_score_for_prompt >= 1.0:
                    break

                beams = [res[1].restart() for res in scored_results[:beam_width]]

            # Populate the enclosed _results dictionary
            _results[initial_prompt] = [best_chat_for_prompt] if best_chat_for_prompt else []
            highest_score_overall = max(highest_score_overall, best_score_for_prompt)

        # Return the single float, as required by the protocol
        return highest_score_overall

    search_callable = t.cast("Search", search)

    return search_callable
