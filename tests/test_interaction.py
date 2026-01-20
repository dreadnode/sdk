from unittest.mock import patch

from dreadnode.agent.tools.interaction import Question, QuestionOption, ask_user


def test_question_model() -> None:
    q = Question(
        id="test",
        question="Test question?",
        options=[
            QuestionOption(label="Option 1", description="First option"),
            QuestionOption(label="Option 2", description="Second option"),
        ],
    )
    assert q.id == "test"
    assert len(q.options) == 2
    assert q.multi_select is False


def test_ask_user_single_choice() -> None:
    questions = [
        {
            "id": "choice",
            "question": "Pick one",
            "options": [
                {"label": "A", "description": "First"},
                {"label": "B", "description": "Second"},
            ],
        }
    ]

    with patch("builtins.input", return_value="1"), patch("builtins.print"):
        result = ask_user(questions)

    assert "choice: A" in result


def test_ask_user_multi_choice() -> None:
    questions = [
        {
            "id": "choices",
            "question": "Pick multiple",
            "options": [
                {"label": "A", "description": "First"},
                {"label": "B", "description": "Second"},
                {"label": "C", "description": "Third"},
            ],
            "multi_select": True,
        }
    ]

    with patch("builtins.input", return_value="1,3"), patch("builtins.print"):
        result = ask_user(questions)

    assert "A" in result
    assert "C" in result
