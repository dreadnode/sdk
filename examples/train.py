"""Training Example - Model Training with dn.train().

This example demonstrates:
- dn.train() - High-level training API
- YAML and dict config options
- Scorers as reward functions
- Dataset loading from HuggingFace

Run with:
    python examples/train.py

Note: Requires GPU and appropriate dependencies (vllm, ray, etc.)
"""

import os

# Set environment before imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import dreadnode as dn


# Define scorers to use as reward functions
@dn.scorer
def has_reasoning(completion: str) -> float:
    """Reward responses that show reasoning steps.

    Looks for keywords that indicate step-by-step thinking.
    """
    keywords = ["step", "first", "then", "next", "because", "therefore", "so"]
    completion_lower = completion.lower()
    matches = sum(1 for k in keywords if k in completion_lower)
    return min(matches / 3, 1.0)


@dn.scorer
def has_answer(completion: str) -> float:
    """Reward responses that have a clear answer.

    Looks for answer patterns like "answer is X" or "= X".
    """
    import re

    patterns = [
        r"(answer|result)\s*(is|=)\s*\d+",
        r"=\s*\d+",
        r"####\s*\d+",
    ]
    completion_lower = completion.lower()
    for pattern in patterns:
        if re.search(pattern, completion_lower):
            return 1.0
    return 0.0


@dn.scorer
def appropriate_length(completion: str) -> float:
    """Reward responses of appropriate length.

    Too short: might be incomplete
    Too long: might be verbose
    """
    length = len(completion)
    if 100 <= length <= 500:
        return 1.0
    elif 50 <= length < 100 or 500 < length <= 800:
        return 0.5
    return 0.2


def train_with_dict_config():
    """Train using a dict configuration."""
    print("\n" + "=" * 50)
    print("Training with dict config and scorers")
    print("=" * 50)

    result = dn.train(
        {
            "trainer": "grpo",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_steps": 50,
            "num_prompts_per_step": 4,
            "num_generations_per_prompt": 4,
            "learning_rate": 1e-6,
            "temperature": 0.7,
            "max_new_tokens": 256,
            "log_interval": 10,
            "checkpoint_interval": 25,
            "checkpoint_dir": "./checkpoints/train_example",
            "dataset": {
                "type": "huggingface",
                "name": "openai/gsm8k",
                "config": "main",
                "split": "train",
                "max_samples": 100,
                "prompt_field": "question",
                "prompt_template": "Solve this math problem step by step:\n\n{question}\n\nAnswer:",
            },
        },
        scorers=[has_reasoning, has_answer, appropriate_length],
    )

    print(f"\nTraining complete!")
    print(f"Result: {result}")
    return result


def train_with_yaml_config():
    """Train using a YAML configuration file."""
    print("\n" + "=" * 50)
    print("Training with YAML config")
    print("=" * 50)

    result = dn.train("examples/configs/grpo_math.yaml")

    print(f"\nTraining complete!")
    print(f"Result: {result}")
    return result


def train_with_custom_prompts():
    """Train with custom prompts and reward function."""
    print("\n" + "=" * 50)
    print("Training with custom prompts")
    print("=" * 50)

    # Custom prompts
    prompts = [
        "What is 5 + 3? Show your work.",
        "Calculate 12 * 4. Explain step by step.",
        "If I have 20 apples and give away 7, how many remain?",
        "What is 100 divided by 4?",
    ]

    # Custom reward function
    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        rewards = []
        for completion in completions:
            # Simple scoring: reward if contains a number
            import re

            has_num = 1.0 if re.search(r"\d+", completion) else 0.0
            has_step = 0.5 if "step" in completion.lower() else 0.0
            rewards.append(min(has_num + has_step, 1.0))
        return rewards

    result = dn.train(
        {
            "trainer": "grpo",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_steps": 20,
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 4,
            "learning_rate": 1e-6,
        },
        prompts=prompts,
        reward_fn=reward_fn,
    )

    print(f"\nTraining complete!")
    print(f"Result: {result}")
    return result


def main():
    print("Training Example")
    print("=" * 50)
    print("\nThis example shows three ways to use dn.train():")
    print("1. Dict config with scorers as rewards")
    print("2. YAML config file")
    print("3. Custom prompts with reward function")
    print("\nNote: Requires GPU and training dependencies.")

    # Uncomment the training method you want to use:

    # Option 1: Dict config with scorers
    train_with_dict_config()

    # Option 2: YAML config
    # train_with_yaml_config()

    # Option 3: Custom prompts
    # train_with_custom_prompts()


if __name__ == "__main__":
    main()
