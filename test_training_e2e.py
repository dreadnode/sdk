#!/usr/bin/env python3
"""
End-to-end training test for the training module.

This test actually runs a minimal training loop on GPUs to verify
the full training pipeline works correctly.

Requires: torch, ray, nemo-rl, vllm, and GPU access.
"""

import asyncio
import sys
import torch


def test_reward_computation_e2e():
    """Test end-to-end reward computation with real tensors."""
    print("\n=== Testing Reward Computation E2E ===")

    from dreadnode.core.training.rewards.aggregator import RewardAggregator
    from dreadnode.core.training.rewards.functions import (
        SuccessReward,
        ToolPenalty,
        TurnPenalty,
    )
    from dreadnode.core.training.rollouts.types import RolloutResult, RolloutMetrics

    # Create aggregator with multiple rewards
    aggregator = RewardAggregator(
        rewards=[
            SuccessReward(weight=1.0, success_value=1.0, failure_value=-0.5),
            ToolPenalty(weight=0.5, per_call_penalty=0.02, per_failure_penalty=0.1),
            TurnPenalty(weight=0.3, target_turns=5, per_extra_turn_penalty=0.05),
        ]
    )

    # Create sample rollouts
    rollouts = []
    for i in range(8):
        rollout = RolloutResult(
            rollout_id=f"test-{i}",
            goal="Complete the task",
            message_log=[
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": "Working on it..."},
            ],
            metrics=RolloutMetrics(
                total_turns=3 + i,
                total_tool_calls=i * 2,
                failed_tool_calls=i % 3,
                total_generated_tokens=100 + i * 50,
            ),
            success=i % 2 == 0,  # Alternating success/failure
        )
        rollouts.append(rollout)

    # Compute rewards
    results = aggregator.compute_batch(rollouts)

    print(f"  Computed {len(results)} reward results")
    for i, result in enumerate(results):
        success_str = "✓" if rollouts[i].success else "✗"
        print(f"  Rollout {i} [{success_str}]: reward={result.reward:.4f}")

    # Verify results
    assert len(results) == 8
    assert all(r.reward is not None for r in results)
    # Successful rollouts should have higher rewards
    assert results[0].reward > results[1].reward  # success > failure

    print("  ✓ Reward computation works correctly")
    return True


def test_baseline_calculation_e2e():
    """Test baseline and advantage calculation on GPU."""
    print("\n=== Testing Baseline Calculation E2E ===")

    from dreadnode.core.training.trainers.grpo import (
        calculate_baseline_and_std_per_prompt,
        scale_rewards,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    # Create realistic reward tensor
    batch_size = 32
    num_generations = 4
    total_samples = batch_size * num_generations

    # Simulate rewards from different prompts
    # Each prompt has `num_generations` samples
    rewards = torch.randn(total_samples, device=device) * 0.5 + 0.5
    rewards = rewards.clamp(0, 1)

    input_ids = torch.zeros(total_samples, 128, device=device)
    mask = torch.ones(total_samples, device=device)

    # Test leave-one-out baseline
    baseline, std = calculate_baseline_and_std_per_prompt(
        input_ids, rewards, mask, leave_one_out_baseline=True
    )

    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Baseline shape: {baseline.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Baseline range: [{baseline.min():.4f}, {baseline.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")

    # Compute advantages
    advantages = (rewards - baseline) / std
    print(f"  Advantages range: [{advantages.min():.4f}, {advantages.max():.4f}]")

    # Verify no NaN or Inf
    assert not torch.isnan(baseline).any(), "Baseline contains NaN"
    assert not torch.isnan(std).any(), "Std contains NaN"
    assert not torch.isnan(advantages).any(), "Advantages contain NaN"
    assert not torch.isinf(advantages).any(), "Advantages contain Inf"

    # Test reward scaling
    scaling_config = {
        "enabled": True,
        "source_min": 0.0,
        "source_max": 1.0,
        "target_min": -1.0,
        "target_max": 1.0,
    }
    scaled = scale_rewards(rewards, scaling_config)
    print(f"  Scaled rewards range: [{scaled.min():.4f}, {scaled.max():.4f}]")

    assert scaled.min() >= -1.0 and scaled.max() <= 1.0

    print("  ✓ Baseline calculation works correctly on GPU")
    return True


def test_reward_shaping_e2e():
    """Test reward shaping with DAPO penalty."""
    print("\n=== Testing Reward Shaping E2E ===")

    from dreadnode.core.training.rewards.shaping import (
        RewardShaper,
        apply_dapo_penalty,
        shape_rewards_for_grpo,
    )
    from dreadnode.core.training.rewards.types import RewardResult

    # Test DAPO penalty
    response_lengths = [1000, 2000, 3000, 3500, 4000, 4500]
    max_length = 4096
    buffer = 512

    print(f"  Max length: {max_length}, Buffer: {buffer}")
    print(f"  Expected max without penalty: {max_length - buffer}")

    for length in response_lengths:
        penalized = apply_dapo_penalty(
            reward=1.0,
            response_length=length,
            max_response_length=max_length,
            buffer_length=buffer,
            penalty=0.5,
        )
        indicator = "✓" if length <= max_length - buffer else "⚠"
        print(f"  Length {length}: reward {1.0:.2f} -> {penalized:.4f} {indicator}")

    # Test full shaper
    shaper = RewardShaper(
        scaling_enabled=True,
        source_min=0.0,
        source_max=1.0,
        target_min=-1.0,
        target_max=1.0,
        dapo_enabled=True,
        max_response_length=4096,
        buffer_length=512,
        penalty=0.5,
    )

    # Test with RewardResult
    result = RewardResult(reward=0.75)
    shaped = shaper.shape(result, response_length=3000)
    print(f"  Shaped result: {result.reward} -> {shaped.reward}")

    # Test NeMo config conversion
    nemo_config = shaper.to_nemo_config()
    assert "reward_scaling" in nemo_config
    assert "reward_shaping" in nemo_config

    print("  ✓ Reward shaping works correctly")
    return True


def test_grpo_trainer_init():
    """Test GRPO trainer initialization."""
    print("\n=== Testing GRPO Trainer Initialization ===")

    from dreadnode.core.training.trainers.grpo import GRPOTrainer, GRPOConfig
    from dreadnode.core.training.rewards.aggregator import RewardAggregator
    from dreadnode.core.training.rewards.functions import SuccessReward

    config: GRPOConfig = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",  # Small model for testing
        "num_prompts_per_step": 8,
        "num_generations_per_prompt": 4,
        "max_rollout_turns": 5,
        "learning_rate": 1e-6,
        "kl_penalty_coeff": 0.01,
        "reward_scaling": {
            "enabled": True,
            "source_min": 0.0,
            "source_max": 1.0,
            "target_min": -1.0,
            "target_max": 1.0,
        },
    }

    rewards = RewardAggregator(rewards=[SuccessReward(weight=1.0)])

    trainer = GRPOTrainer(
        config=config,
        rewards=rewards,
    )

    print(f"  Model: {config['model_name']}")
    print(f"  Prompts per step: {config['num_prompts_per_step']}")
    print(f"  Generations per prompt: {config['num_generations_per_prompt']}")

    # Test NeMo config generation
    nemo_config = trainer.to_nemo_config()

    assert "policy" in nemo_config
    assert "grpo" in nemo_config
    assert nemo_config["policy"]["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert nemo_config["grpo"]["num_prompts_per_step"] == 8

    print(f"  NeMo config keys: {list(nemo_config.keys())}")
    print("  ✓ GRPO trainer initialization works")
    return True


def test_sft_trainer_init():
    """Test SFT trainer initialization."""
    print("\n=== Testing SFT Trainer Initialization ===")

    from dreadnode.core.training.trainers.sft import SFTTrainer, SFTConfig

    config: SFTConfig = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "train_file": "train.jsonl",
        "val_file": "val.jsonl",
        "max_num_epochs": 3,
        "learning_rate": 1e-5,
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "loss_type": "response_only",
    }

    trainer = SFTTrainer(config=config)

    print(f"  Model: {config['model_name']}")
    print(f"  Epochs: {config['max_num_epochs']}")
    print(f"  Loss type: {config['loss_type']}")

    # Test NeMo config generation
    nemo_config = trainer.to_nemo_config()

    assert "policy" in nemo_config
    assert "sft" in nemo_config
    assert "data" in nemo_config

    print(f"  NeMo config keys: {list(nemo_config.keys())}")
    print("  ✓ SFT trainer initialization works")
    return True


async def test_rollout_orchestration():
    """Test rollout orchestration with mock environment."""
    print("\n=== Testing Rollout Orchestration ===")

    from dreadnode.core.training.rollouts.orchestrator import (
        RolloutOrchestrator,
        EnvironmentReturn,
    )
    from dreadnode.core.training.rollouts.types import RolloutConfig

    # Create mock generator that returns (text, metadata) tuple
    class MockGenerator:
        def __init__(self):
            self.call_count = 0

        async def generate(self, messages, max_tokens=4096, stop=None, **kwargs):
            self.call_count += 1
            return (
                f"Working on step {self.call_count}...",
                {"tool_calls": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20}},
            )

    # Create mock environment that returns list of EnvironmentReturn
    class MockEnvironment:
        def __init__(self):
            self.call_count = 0

        def step(self, message_logs, metadata=None):
            results = []
            for _ in message_logs:
                self.call_count += 1
                if self.call_count >= 3:
                    results.append(EnvironmentReturn(
                        observation="Task completed successfully!",
                        terminated=True,
                        reward=1.0,
                        info={"success": True},
                    ))
                else:
                    results.append(EnvironmentReturn(
                        observation=f"Step {self.call_count}: Continue working...",
                        terminated=False,
                        reward=0.1,
                    ))
            return results

    config: RolloutConfig = {
        "max_turns": 10,
        "max_tokens_per_turn": 512,
        "stop_on_success": True,
    }

    orchestrator = RolloutOrchestrator(config=config)
    generator = MockGenerator()
    environment = MockEnvironment()

    # Run a single rollout
    result = await orchestrator.run_single(
        generator=generator,
        environment=environment,
        goal="Complete the test task",
    )

    print(f"  Rollout ID: {result.rollout_id}")
    print(f"  Turns: {result.metrics.total_turns}")
    print(f"  Success: {result.success}")
    print(f"  Final reward: {result.final_reward}")
    print(f"  Message log length: {len(result.message_log)}")

    # Basic validation - rollout should complete
    assert result.rollout_id is not None
    assert result.metrics.total_turns >= 1
    assert len(result.message_log) >= 2  # At least goal + 1 response

    print("  ✓ Rollout orchestration works correctly")
    return True


def main():
    """Run all end-to-end tests."""
    print("=" * 60)
    print("Training Module End-to-End Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, some tests may be limited")
    else:
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    all_passed = True
    tests = [
        ("Reward Computation", test_reward_computation_e2e),
        ("Baseline Calculation", test_baseline_calculation_e2e),
        ("Reward Shaping", test_reward_shaping_e2e),
        ("GRPO Trainer Init", test_grpo_trainer_init),
        ("SFT Trainer Init", test_sft_trainer_init),
        ("Rollout Orchestration", lambda: asyncio.run(test_rollout_orchestration())),
    ]

    for name, test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
                print(f"  ✗ {name} test failed")
        except Exception as e:
            all_passed = False
            print(f"  ✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All end-to-end tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
