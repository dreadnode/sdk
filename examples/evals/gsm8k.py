import rigging as rg
from datasets import load_dataset

import dreadnode as dn


class Answer(rg.Model):
    reasoning: str = rg.element(description="Your reasoning.")
    final_answer: float = rg.element(description="Single float value.")


@dn.task
async def solve_math_problem(question: str, model: str, guidance: str = "") -> float:
    @rg.prompt(generator_id=model)
    async def answer_question(question: str, guidance: str) -> Answer:
        "Answer the following math question."

    answer = await answer_question.set_(max_parsing_rounds=0)(question, guidance)
    dn.log_output("reasoning", answer.reasoning)
    return answer.final_answer


def prepare_gsm8k(row: dict) -> dict:
    reasoning, answer = row["answer"].split("####")
    return {
        "question": row["question"],
        "reasoning": reasoning.strip(),
        "answer": float(answer.strip()),
    }


gsm8k_dataset = (
    load_dataset("gsm8k", "main", split="test").select(range(10)).map(prepare_gsm8k).to_list()
)

# Define the evaluation

gsm8k_eval = solve_math_problem.as_eval(
    name="GSM8K",
    dataset=gsm8k_dataset,
    parameters={
        "model": ["gpt-4o-mini", "claude-3-5-haiku-latest"],
    },
    scorers={
        "correct": dn.scorers.equals(dn.DatasetField("answer")),
        "similarity": dn.scorers.similarity(dn.DatasetField("reasoning")),
    },
    assert_scores=["correct"],
    concurrency=3,
)


# Run the evaluation

result = await gsm8k_eval.console()
