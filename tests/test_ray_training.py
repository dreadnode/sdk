"""
Unit tests for the Ray-native training module.

Tests cover:
- Experience data structures (Experience, ExperienceBatch, ExperienceBuffer)
- RolloutEnvironment protocol
- Sequence packing for SFT
- GRPO advantage computation
- Configuration classes

Note: Tests that require GPU/Ray are marked to skip gracefully.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if ray is available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Skip markers
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch is not installed"
)

requires_ray = pytest.mark.skipif(
    not RAY_AVAILABLE or not TORCH_AVAILABLE,
    reason="ray or torch is not installed"
)


# =============================================================================
# Experience Module Tests
# =============================================================================


@requires_torch
class TestExperience:
    """Tests for Experience dataclass."""

    def test_creation(self):
        """Test basic Experience creation."""
        from dreadnode.core.training.ray.experience import Experience

        exp = Experience(
            prompt="What is 2+2?",
            completion="4",
            prompt_ids=torch.tensor([1, 2, 3]),
            completion_ids=torch.tensor([4]),
            reward=1.0,
        )

        assert exp.prompt == "What is 2+2?"
        assert exp.completion == "4"
        assert exp.reward == 1.0

    def test_auto_full_ids(self):
        """Test automatic full_ids computation."""
        from dreadnode.core.training.ray.experience import Experience

        prompt_ids = torch.tensor([1, 2, 3])
        completion_ids = torch.tensor([4, 5])

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            reward=0.5,
        )

        assert exp.full_ids is not None
        assert exp.full_ids.tolist() == [1, 2, 3, 4, 5]

    def test_auto_completion_mask(self):
        """Test automatic completion_mask computation."""
        from dreadnode.core.training.ray.experience import Experience

        prompt_ids = torch.tensor([1, 2, 3])
        completion_ids = torch.tensor([4, 5])

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            reward=0.5,
        )

        assert exp.completion_mask is not None
        # First 3 tokens are prompt (False), last 2 are completion (True)
        assert exp.completion_mask.tolist() == [False, False, False, True, True]

    def test_to_device(self):
        """Test moving tensors to device."""
        from dreadnode.core.training.ray.experience import Experience

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=torch.tensor([1, 2, 3]),
            completion_ids=torch.tensor([4, 5]),
            logprobs=torch.tensor([0.1, 0.2]),
            reward=0.5,
        )

        # Move to CPU (safe for testing without GPU)
        exp_cpu = exp.to_device("cpu")

        assert exp_cpu.prompt_ids.device.type == "cpu"
        assert exp_cpu.logprobs.device.type == "cpu"


@requires_torch
class TestExperienceBatch:
    """Tests for ExperienceBatch."""

    def test_creation(self):
        """Test basic ExperienceBatch creation."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        exp1 = Experience(
            prompt="Q1",
            completion="A1",
            prompt_ids=torch.tensor([1, 2]),
            completion_ids=torch.tensor([3]),
            reward=1.0,
            group_id=0,
        )
        exp2 = Experience(
            prompt="Q1",
            completion="A2",
            prompt_ids=torch.tensor([1, 2]),
            completion_ids=torch.tensor([4]),
            reward=0.5,
            group_id=0,
        )

        batch = ExperienceBatch(experiences=[exp1, exp2])

        assert len(batch) == 2
        assert not batch.is_tensorized

    def test_compute_advantages_leave_one_out(self):
        """Test GRPO leave-one-out advantage computation."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        # Create 4 experiences with same group_id
        experiences = [
            Experience(
                prompt="Q",
                completion=f"A{i}",
                prompt_ids=torch.tensor([1]),
                completion_ids=torch.tensor([i]),
                reward=float(i),  # rewards: 0, 1, 2, 3
                group_id=0,
            )
            for i in range(4)
        ]

        batch = ExperienceBatch(experiences=experiences)
        batch.compute_advantages(use_leave_one_out=True)

        # Leave-one-out baseline:
        # For reward=0: baseline = mean([1,2,3]) = 2.0, advantage = 0 - 2 = -2
        # For reward=1: baseline = mean([0,2,3]) = 5/3, advantage = 1 - 5/3 = -2/3
        # For reward=2: baseline = mean([0,1,3]) = 4/3, advantage = 2 - 4/3 = 2/3
        # For reward=3: baseline = mean([0,1,2]) = 1.0, advantage = 3 - 1 = 2
        assert experiences[0].advantage == pytest.approx(-2.0, rel=0.01)
        assert experiences[1].advantage == pytest.approx(-2/3, rel=0.01)
        assert experiences[2].advantage == pytest.approx(2/3, rel=0.01)
        assert experiences[3].advantage == pytest.approx(2.0, rel=0.01)

    def test_compute_advantages_standard(self):
        """Test standard (non-leave-one-out) advantage computation."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        experiences = [
            Experience(
                prompt="Q",
                completion=f"A{i}",
                prompt_ids=torch.tensor([1]),
                completion_ids=torch.tensor([i]),
                reward=float(i),  # rewards: 0, 1, 2, 3
                group_id=0,
            )
            for i in range(4)
        ]

        batch = ExperienceBatch(experiences=experiences)
        batch.compute_advantages(use_leave_one_out=False)

        # Standard baseline: mean([0,1,2,3]) = 1.5 for all
        assert experiences[0].advantage == pytest.approx(-1.5, rel=0.01)
        assert experiences[1].advantage == pytest.approx(-0.5, rel=0.01)
        assert experiences[2].advantage == pytest.approx(0.5, rel=0.01)
        assert experiences[3].advantage == pytest.approx(1.5, rel=0.01)

    def test_merge_batches(self):
        """Test merging multiple batches."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        batch1 = ExperienceBatch(experiences=[
            Experience(
                prompt="Q1",
                completion="A1",
                prompt_ids=torch.tensor([1]),
                completion_ids=torch.tensor([2]),
                reward=1.0,
            )
        ])
        batch2 = ExperienceBatch(experiences=[
            Experience(
                prompt="Q2",
                completion="A2",
                prompt_ids=torch.tensor([3]),
                completion_ids=torch.tensor([4]),
                reward=0.5,
            )
        ])

        merged = ExperienceBatch.merge([batch1, batch2])

        assert len(merged) == 2

    def test_to_tensors(self):
        """Test tensorization of batch."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch

        experiences = [
            Experience(
                prompt="Q",
                completion="A1",
                prompt_ids=torch.tensor([1, 2]),
                completion_ids=torch.tensor([3]),
                reward=1.0,
                advantage=0.5,
            ),
            Experience(
                prompt="Q",
                completion="A2",
                prompt_ids=torch.tensor([1, 2]),
                completion_ids=torch.tensor([4, 5]),
                reward=0.5,
                advantage=-0.5,
            ),
        ]

        batch = ExperienceBatch(experiences=experiences)

        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0

        tensorized = batch.to_tensors(tokenizer, device="cpu")

        assert tensorized.is_tensorized
        assert tensorized.input_ids is not None
        assert tensorized.input_ids.shape[0] == 2  # batch size
        assert tensorized.advantages.shape == (2,)


@requires_torch
class TestExperienceBuffer:
    """Tests for ExperienceBuffer async queue."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        """Test basic put and get operations."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch, ExperienceBuffer

        buffer = ExperienceBuffer(max_size=2)

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=torch.tensor([1]),
            completion_ids=torch.tensor([2]),
            reward=1.0,
        )
        batch = ExperienceBatch(experiences=[exp])

        await buffer.put(batch)
        assert buffer.size == 1

        retrieved = await buffer.get()
        assert len(retrieved) == 1
        assert buffer.size == 0

    @pytest.mark.asyncio
    async def test_try_put_full_buffer(self):
        """Test non-blocking put when buffer is full."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch, ExperienceBuffer

        buffer = ExperienceBuffer(max_size=1)

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=torch.tensor([1]),
            completion_ids=torch.tensor([2]),
            reward=1.0,
        )
        batch = ExperienceBatch(experiences=[exp])

        # First put should succeed
        assert buffer.try_put(batch) is True
        assert buffer.is_full

        # Second put should fail (buffer full)
        assert buffer.try_put(batch) is False
        assert buffer.stats["dropped"] == 1

    @pytest.mark.asyncio
    async def test_stats(self):
        """Test buffer statistics."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch, ExperienceBuffer

        buffer = ExperienceBuffer(max_size=5)

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=torch.tensor([1]),
            completion_ids=torch.tensor([2]),
            reward=1.0,
        )
        batch = ExperienceBatch(experiences=[exp])

        await buffer.put(batch)
        await buffer.put(batch)
        await buffer.get()

        stats = buffer.stats
        assert stats["produced"] == 2
        assert stats["consumed"] == 1
        assert stats["current_size"] == 1

    @pytest.mark.asyncio
    async def test_drain(self):
        """Test draining all batches."""
        from dreadnode.core.training.ray.experience import Experience, ExperienceBatch, ExperienceBuffer

        buffer = ExperienceBuffer(max_size=5)

        exp = Experience(
            prompt="test",
            completion="test",
            prompt_ids=torch.tensor([1]),
            completion_ids=torch.tensor([2]),
            reward=1.0,
        )
        batch = ExperienceBatch(experiences=[exp])

        for _ in range(3):
            await buffer.put(batch)

        drained = await buffer.drain()

        assert len(drained) == 3
        assert buffer.is_empty


# =============================================================================
# RolloutEnvironment Protocol Tests
# =============================================================================


@requires_torch
class TestRolloutEnvironment:
    """Tests for RolloutEnvironment protocol."""

    def test_protocol_check(self):
        """Test that protocol is runtime checkable."""
        from dreadnode.core.training.ray.rollout_env import RolloutEnvironment

        # A class that implements the protocol
        class MockRolloutEnv:
            async def generate_experiences(self, prompts, num_generations_per_prompt=4):
                return []

            async def generate_batch(self, prompts, num_generations_per_prompt=4):
                from dreadnode.core.training.ray.experience import ExperienceBatch
                return ExperienceBatch(experiences=[])

            def update_weights(self, state_dict):
                pass

        env = MockRolloutEnv()
        assert isinstance(env, RolloutEnvironment)


# =============================================================================
# SFT Module Tests
# =============================================================================


@requires_torch
class TestSequencePacker:
    """Tests for SequencePacker."""

    def test_pack_single_sequence(self):
        """Test packing a single sequence."""
        from dreadnode.core.training.ray.sft import SequencePacker

        packer = SequencePacker(max_seq_length=10, pad_token_id=0)

        sequences = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}
        ]

        packed = packer.pack_sequences(sequences)

        assert len(packed) == 1
        assert packed[0].input_ids.shape[0] == 10  # Padded to max_seq_length
        assert packed[0].input_ids[:3].tolist() == [1, 2, 3]
        assert packed[0].input_ids[3:].tolist() == [0] * 7  # Padding

    def test_pack_multiple_sequences(self):
        """Test packing multiple sequences into one sample."""
        from dreadnode.core.training.ray.sft import SequencePacker

        packer = SequencePacker(max_seq_length=20, pad_token_id=0)

        sequences = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
            {"input_ids": torch.tensor([6, 7, 8, 9]), "labels": torch.tensor([6, 7, 8, 9])},
        ]

        packed = packer.pack_sequences(sequences)

        # All should fit in one packed sample (3 + 2 + 4 = 9 < 20)
        assert len(packed) == 1
        # Check the packed sequence
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert packed[0].input_ids[:9].tolist() == expected

    def test_pack_overflow(self):
        """Test that sequences overflow to new packed samples."""
        from dreadnode.core.training.ray.sft import SequencePacker

        packer = SequencePacker(max_seq_length=5, pad_token_id=0)

        sequences = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
        ]

        packed = packer.pack_sequences(sequences)

        # First sequence (3) fits, second (3) overflows (3+3=6 > 5)
        assert len(packed) == 2

    def test_sample_boundaries(self):
        """Test that sample boundaries are tracked correctly."""
        from dreadnode.core.training.ray.sft import SequencePacker

        packer = SequencePacker(max_seq_length=20, pad_token_id=0)

        sequences = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        ]

        packed = packer.pack_sequences(sequences)

        # Boundaries: [0, 3, 5]
        assert packed[0].sample_boundaries == [0, 3, 5]


@requires_torch
class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.sft import SFTConfig

        config = SFTConfig()

        assert config.max_seq_length == 2048
        assert config.use_packing is True
        assert config.learning_rate == 2e-5

    def test_tokenizer_name_default(self):
        """Test tokenizer_name defaults to model_name."""
        from dreadnode.core.training.ray.sft import SFTConfig

        config = SFTConfig(model_name="test-model")

        assert config.tokenizer_name == "test-model"


# =============================================================================
# Configuration Tests
# =============================================================================


@requires_torch
class TestRayGRPOConfig:
    """Tests for RayGRPOConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.config import RayGRPOConfig

        config = RayGRPOConfig()

        assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.num_generations_per_prompt == 4
        assert config.learning_rate == 1e-6

    def test_train_batch_size(self):
        """Test train_batch_size property."""
        from dreadnode.core.training.ray.config import RayGRPOConfig

        config = RayGRPOConfig(
            num_prompts_per_step=8,
            num_generations_per_prompt=4,
        )

        assert config.train_batch_size == 32  # 8 * 4

    def test_to_dict(self):
        """Test serialization to dict."""
        from dreadnode.core.training.ray.config import RayGRPOConfig

        config = RayGRPOConfig(model_name="test-model")
        d = config.to_dict()

        assert d["model_name"] == "test-model"
        assert "vllm" in d
        assert "training" in d
        assert "loss" in d


@requires_torch
class TestFSDP2Config:
    """Tests for FSDP2Config."""

    def test_default_values(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config

        config = FSDP2Config()

        assert config.sharding_strategy == "full"
        assert config.param_dtype == torch.bfloat16
        assert config.reduce_dtype == torch.float32
        assert config.use_activation_checkpointing is True


@requires_torch
class TestAsyncGRPOConfig:
    """Tests for AsyncGRPOConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.coordinator import AsyncGRPOConfig

        config = AsyncGRPOConfig()

        assert config.num_rollout_workers == 1
        assert config.buffer_size == 1
        assert config.weight_sync_interval == 1


# =============================================================================
# Learner Tests (without GPU)
# =============================================================================


@requires_torch
class TestGRPOLoss:
    """Tests for GRPO loss computation."""

    def test_compute_log_probs(self):
        """Test log probability computation."""
        from dreadnode.core.training.ray.learner import compute_log_probs

        # Mock logits [batch=2, seq=4, vocab=10]
        logits = torch.randn(2, 4, 10)
        input_ids = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        log_probs = compute_log_probs(logits, input_ids, mask)

        # Output should be [batch=2, seq=3] (shifted)
        assert log_probs.shape == (2, 3)

    def test_grpo_loss_shape(self):
        """Test GRPO loss output shape."""
        from dreadnode.core.training.ray.learner import grpo_loss, TrainingBatch
        from dreadnode.core.training.ray.config import GRPOLossConfig

        batch_size = 4
        seq_len = 16
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        batch = TrainingBatch(
            input_ids=torch.randint(0, vocab_size, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            completion_mask=torch.ones(batch_size, seq_len),
            advantages=torch.randn(batch_size),
            generation_logprobs=torch.randn(batch_size, seq_len),
        )
        config = GRPOLossConfig()

        loss, metrics = grpo_loss(logits, batch, config)

        assert loss.shape == ()  # Scalar
        assert "loss" in metrics
        assert "pg_loss" in metrics


# =============================================================================
# Training State Tests
# =============================================================================


@requires_torch
class TestTrainingState:
    """Tests for TrainingState."""

    def test_creation(self):
        """Test basic TrainingState creation."""
        from dreadnode.core.training.ray.coordinator import TrainingState

        state = TrainingState()

        assert state.step == 0
        assert state.samples_seen == 0
        assert state.best_reward == float("-inf")

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        from dreadnode.core.training.ray.coordinator import TrainingState
        from datetime import datetime
        import time

        state = TrainingState()
        state.started_at = datetime.now()

        time.sleep(0.1)

        elapsed = state.elapsed_seconds()
        assert elapsed >= 0.1

    def test_samples_per_second(self):
        """Test throughput calculation."""
        from dreadnode.core.training.ray.coordinator import TrainingState
        from datetime import datetime, timedelta

        state = TrainingState()
        state.started_at = datetime.now() - timedelta(seconds=10)
        state.samples_seen = 100

        throughput = state.samples_per_second()
        assert throughput == pytest.approx(10.0, rel=0.1)


# =============================================================================
# RolloutConfig Tests
# =============================================================================


@requires_torch
class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_from_grpo_config(self):
        """Test creation from RayGRPOConfig."""
        from dreadnode.core.training.ray.config import RayGRPOConfig
        from dreadnode.core.training.ray.rollout_worker import RolloutConfig

        grpo_config = RayGRPOConfig(
            model_name="test-model",
            temperature=0.8,
            num_generations_per_prompt=8,
        )

        rollout_config = RolloutConfig.from_grpo_config(grpo_config)

        assert rollout_config.model_name == "test-model"
        assert rollout_config.temperature == 0.8
        assert rollout_config.num_generations_per_prompt == 8


# =============================================================================
# Integration-style Tests (mocked)
# =============================================================================


@requires_torch
class TestAsyncGRPOCoordinatorMocked:
    """Mocked tests for AsyncGRPOCoordinator (no Ray/GPU required)."""

    def test_get_prompt_batch(self):
        """Test prompt batching logic."""
        from dreadnode.core.training.ray.coordinator import AsyncGRPOCoordinator

        # Create coordinator with mocked components
        with patch('dreadnode.core.training.ray.coordinator.ExperienceBuffer'):
            with patch('dreadnode.core.training.ray.coordinator.RolloutWorkerPool'):
                with patch('dreadnode.core.training.ray.coordinator.FSDP2Learner'):
                    from dreadnode.core.training.ray.config import RayGRPOConfig

                    config = RayGRPOConfig()
                    prompts = [f"prompt_{i}" for i in range(10)]

                    coordinator = AsyncGRPOCoordinator.__new__(AsyncGRPOCoordinator)
                    coordinator.prompts = prompts
                    coordinator.async_config = Mock()
                    coordinator.async_config.prompts_per_step = 3

                    # Test batch retrieval
                    batch = coordinator._get_prompt_batch(0)
                    assert batch == ["prompt_0", "prompt_1", "prompt_2"]

                    # Test wrap-around
                    batch = coordinator._get_prompt_batch(8)
                    assert batch == ["prompt_8", "prompt_9", "prompt_0"]


# =============================================================================
# Distributed Config Tests
# =============================================================================


@requires_torch
class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.distributed import DistributedConfig

        config = DistributedConfig()

        assert config.num_workers == 1
        assert config.use_gpu is True
        assert config.max_failures == 3
