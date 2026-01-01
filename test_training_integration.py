#!/usr/bin/env python3
"""
Test script for training module integration with NeMo RL.

Run this on a machine with torch, ray, and nemo-rl installed.
"""

import asyncio
import sys


def test_imports():
    """Test all training module imports."""
    print("Testing imports...")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import ray
        print(f"  ✓ ray {ray.__version__}")
    except ImportError as e:
        print(f"  ✗ ray: {e}")
        return False

    try:
        import nemo_rl
        print(f"  ✓ nemo_rl")
    except ImportError as e:
        print(f"  ✗ nemo_rl: {e}")
        return False

    # Now test training module imports
    try:
        from dreadnode.core.training.trainers.grpo import (
            GRPOTrainer,
            GRPOConfig,
            scale_rewards,
            calculate_baseline_and_std_per_prompt,
        )
        print("  ✓ GRPOTrainer imports")
    except ImportError as e:
        print(f"  ✗ GRPOTrainer: {e}")
        return False

    try:
        from dreadnode.core.training.trainers.sft import SFTTrainer, SFTConfig
        print("  ✓ SFTTrainer imports")
    except ImportError as e:
        print(f"  ✗ SFTTrainer: {e}")
        return False

    try:
        from dreadnode.core.training.rewards import (
            RewardAggregator,
            SuccessReward,
            ToolPenalty,
        )
        print("  ✓ Rewards imports")
    except ImportError as e:
        print(f"  ✗ Rewards: {e}")
        return False

    try:
        from dreadnode.core.training.rollouts import (
            RolloutOrchestrator,
            RolloutConfig,
            DNAgentAdapter,
        )
        print("  ✓ Rollouts imports")
    except ImportError as e:
        print(f"  ✗ Rollouts: {e}")
        return False

    return True


def test_grpo_config():
    """Test GRPOConfig and to_nemo_config conversion."""
    print("\nTesting GRPOConfig...")

    from dreadnode.core.training.trainers.grpo import GRPOTrainer, GRPOConfig

    config: GRPOConfig = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "num_prompts_per_step": 32,
        "num_generations_per_prompt": 4,
        "max_rollout_turns": 10,
        "learning_rate": 1e-5,
        "kl_penalty_coeff": 0.01,
        "reward_scaling": {
            "enabled": True,
            "source_min": 0.0,
            "source_max": 1.0,
            "target_min": -1.0,
            "target_max": 1.0,
        },
    }

    trainer = GRPOTrainer(config=config)
    nemo_config = trainer.to_nemo_config()

    assert nemo_config["policy"]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert nemo_config["grpo"]["num_prompts_per_step"] == 32
    assert nemo_config["grpo"]["kl_penalty_coeff"] == 0.01
    assert nemo_config["grpo"]["reward_scaling"]["enabled"] == True

    print("  ✓ GRPOConfig works")
    print("  ✓ to_nemo_config works")
    return True


def test_sft_config():
    """Test SFTConfig and to_nemo_config conversion."""
    print("\nTesting SFTConfig...")

    from dreadnode.core.training.trainers.sft import SFTTrainer, SFTConfig

    config: SFTConfig = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "train_file": "train.jsonl",
        "val_file": "val.jsonl",
        "max_num_epochs": 3,
        "learning_rate": 1e-5,
        "loss_type": "response_only",
    }

    trainer = SFTTrainer(config=config)
    nemo_config = trainer.to_nemo_config()

    assert nemo_config["policy"]["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert nemo_config["sft"]["max_num_epochs"] == 3
    assert nemo_config["data"]["train_file"] == "train.jsonl"

    print("  ✓ SFTConfig works")
    print("  ✓ to_nemo_config works")
    return True


def test_reward_functions():
    """Test reward functions."""
    print("\nTesting reward functions...")

    import torch
    from dreadnode.core.training.trainers.grpo import (
        scale_rewards,
        calculate_baseline_and_std_per_prompt,
    )

    # Test scale_rewards
    rewards = torch.tensor([0.0, 0.5, 1.0])
    config = {
        "enabled": True,
        "source_min": 0.0,
        "source_max": 1.0,
        "target_min": -1.0,
        "target_max": 1.0,
    }
    scaled = scale_rewards(rewards, config)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(scaled, expected), f"Expected {expected}, got {scaled}"
    print("  ✓ scale_rewards works")

    # Test baseline calculation
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    input_ids = torch.zeros(4, 10)
    mask = torch.ones(4)

    baseline, std = calculate_baseline_and_std_per_prompt(
        input_ids, rewards, mask, leave_one_out_baseline=True
    )

    # Leave-one-out baseline: each baseline should be mean of others
    # For [1,2,3,4]: baseline[0] = (2+3+4)/3 = 3.0
    assert baseline[0].item() == 3.0, f"Expected 3.0, got {baseline[0].item()}"
    print("  ✓ calculate_baseline_and_std_per_prompt works")

    return True


def test_reward_aggregator():
    """Test reward aggregator."""
    print("\nTesting RewardAggregator...")

    from dreadnode.core.training.rewards import (
        RewardAggregator,
        SuccessReward,
        ToolPenalty,
    )

    aggregator = RewardAggregator(
        rewards=[
            SuccessReward(weight=1.0),
            ToolPenalty(weight=0.1, per_call_penalty=0.05),
        ]
    )

    print(f"  ✓ RewardAggregator created with {len(aggregator.rewards)} rewards")
    return True


async def test_trainer_initialization():
    """Test trainer can be initialized (without actually training)."""
    print("\nTesting trainer initialization...")

    from dreadnode.core.training.trainers.grpo import GRPOTrainer
    from dreadnode.core.training.rewards import RewardAggregator, SuccessReward

    rewards = RewardAggregator(rewards=[SuccessReward(weight=1.0)])

    trainer = GRPOTrainer(
        rewards=rewards,
        config={
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "num_prompts_per_step": 4,
            "num_generations_per_prompt": 2,
        },
    )

    print("  ✓ GRPOTrainer initialized")

    # Check that to_nemo_config works
    nemo_config = trainer.to_nemo_config()
    assert "policy" in nemo_config
    assert "grpo" in nemo_config
    print("  ✓ to_nemo_config produces valid config")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Training Module Integration Tests")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return 1

    # Test configs
    try:
        if not test_grpo_config():
            all_passed = False
    except Exception as e:
        print(f"  ✗ GRPOConfig test failed: {e}")
        all_passed = False

    try:
        if not test_sft_config():
            all_passed = False
    except Exception as e:
        print(f"  ✗ SFTConfig test failed: {e}")
        all_passed = False

    # Test reward functions
    try:
        if not test_reward_functions():
            all_passed = False
    except Exception as e:
        print(f"  ✗ Reward functions test failed: {e}")
        all_passed = False

    # Test reward aggregator
    try:
        if not test_reward_aggregator():
            all_passed = False
    except Exception as e:
        print(f"  ✗ Reward aggregator test failed: {e}")
        all_passed = False

    # Test trainer initialization
    try:
        if not asyncio.run(test_trainer_initialization()):
            all_passed = False
    except Exception as e:
        print(f"  ✗ Trainer initialization test failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
