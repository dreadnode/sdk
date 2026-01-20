import typing as t

from pydantic import BaseModel, Field

from dreadnode.agent.tools.base import tool


class QuestionOption(BaseModel):
    label: str = Field(..., description="Display text for this option")
    description: str = Field(..., description="Explanation of this option")


class Question(BaseModel):
    id: str = Field(..., description="Unique identifier for this question")
    question: str = Field(..., description="The question text")
    options: list[QuestionOption] = Field(..., min_length=2, max_length=4)
    multi_select: bool = Field(default=False)


@tool(catch=True)
def ask_user(
    questions: t.Annotated[
        list[dict[str, t.Any]],
        "List of questions to ask. Each has: id, question, options (list of {label, description}), multi_select (optional)",
    ],
) -> str:
    """
    Ask the user one or more multiple-choice questions during execution.

    Use this when you need user input to make decisions, clarify requirements,
    or get preferences that affect your work.

    Each question should have 2-4 options. Users can select one option, or multiple
    if multi_select is true.

    Returns a formatted string with the user's responses.
    """
    from dreadnode import log_metric, log_output

    parsed = [Question(**q) for q in questions]

    print("\n" + "=" * 80)
    print("Agent needs your input:")
    print("=" * 80 + "\n")

    responses: dict[str, str | list[str]] = {}

    for q_num, q in enumerate(parsed, 1):
        print(f"Question {q_num}/{len(parsed)}: {q.question}\n")

        for idx, opt in enumerate(q.options, 1):
            print(f"  {idx}. {opt.label}")
            print(f"     {opt.description}\n")

        if q.multi_select:
            choice_str = input("Your choice(s) (comma-separated): ").strip()
            choices = [int(c.strip()) for c in choice_str.split(",")]
            selected: list[str] = [q.options[i - 1].label for i in choices]
            responses[q.id] = selected
        else:
            choice = int(input("Your choice: ").strip())
            selected_option: str = q.options[choice - 1].label
            responses[q.id] = selected_option

        print()

    print("=" * 80 + "\n")

    log_output("user_questions", [q.model_dump() for q in parsed])
    log_output("user_responses", responses)
    log_metric("questions_asked", len(parsed))

    result = "User responses:\n"
    for q_id, answer in responses.items():
        if isinstance(answer, list):
            result += f"{q_id}: {', '.join(answer)}\n"
        else:
            result += f"{q_id}: {answer}\n"

    return result
