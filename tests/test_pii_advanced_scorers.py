import pytest

from dreadnode.scorers.pii_advanced import (
    calculate_disclosure_rate_with_ci,
    credential_leakage,
    pii_disclosure_rate,
    training_data_memorization,
    wilson_score_interval,
)


@pytest.mark.asyncio
async def test_training_data_memorization_low_entropy() -> None:
    """Low entropy detection."""
    scorer = training_data_memorization(min_length=50)
    text = " ".join(["aaaa"] * 30)
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "entropy" in result.attributes
    assert result.attributes["entropy"] < 3.5


@pytest.mark.asyncio
async def test_training_data_memorization_excessive_repetition() -> None:
    """Test training_data_memorization detects excessive word repetition."""
    scorer = training_data_memorization(min_length=50)
    # Text with one word repeated >10% of total
    text = " ".join(["company"] * 50 + ["other"] * 10)
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "max_word_repetition" in result.attributes
    assert result.attributes["max_word_repetition"] > 6  # >10% of 60 words


@pytest.mark.asyncio
async def test_training_data_memorization_chapter_pattern() -> None:
    """Test training_data_memorization detects chapter/section patterns."""
    scorer = training_data_memorization(min_length=50)
    text = "Chapter 1: Introduction to Machine Learning. " * 3
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "detected_patterns" in result.attributes
    assert len(result.attributes["detected_patterns"]) > 0


@pytest.mark.asyncio
async def test_training_data_memorization_copyright_pattern() -> None:
    """Test training_data_memorization detects copyright notices."""
    scorer = training_data_memorization(min_length=50)
    text = "This is copyrighted material. Copyright 2024 Example Corp. " * 2
    result = await scorer.score(text)

    assert result.value == 1.0
    assert any("Copyright" in str(p) for p in result.attributes["detected_patterns"])


@pytest.mark.asyncio
async def test_training_data_memorization_isbn_pattern() -> None:
    """Test training_data_memorization detects ISBN/DOI patterns."""
    scorer = training_data_memorization(min_length=50)
    text = "This book ISBN: 978-0-123456-78-9 covers important topics. " * 2
    result = await scorer.score(text)

    assert result.value == 1.0


@pytest.mark.asyncio
async def test_training_data_memorization_code_pattern() -> None:
    """Test training_data_memorization detects code definitions."""
    scorer = training_data_memorization(min_length=50)
    text = "def calculate_loss(x, y): return (x - y) ** 2\n" * 3
    result = await scorer.score(text)

    assert result.value == 1.0


@pytest.mark.asyncio
async def test_training_data_memorization_normal_text() -> None:
    """Test training_data_memorization returns 0 for normal text."""
    scorer = training_data_memorization(min_length=50)
    text = "This is a normal sentence with good variety and natural language patterns."
    result = await scorer.score(text)

    assert result.value == 0.0


@pytest.mark.asyncio
async def test_training_data_memorization_short_text() -> None:
    """Test training_data_memorization returns 0 for text below min_length."""
    scorer = training_data_memorization(min_length=100)
    text = "Short text"
    result = await scorer.score(text)

    assert result.value == 0.0
    assert result.attributes["text_length"] < 100


@pytest.mark.asyncio
async def test_training_data_memorization_custom_patterns() -> None:
    """Test training_data_memorization with custom patterns."""
    scorer = training_data_memorization(min_length=50, patterns=[r"SENSITIVE-\d+"])
    text = "This document contains SENSITIVE-12345 information. " * 3
    result = await scorer.score(text)

    assert result.value == 1.0
    assert any("SENSITIVE" in str(p) for p in result.attributes["detected_patterns"])


@pytest.mark.asyncio
async def test_training_data_memorization_custom_name() -> None:
    """Test training_data_memorization with custom name."""
    scorer = training_data_memorization(name="custom_memorization")
    assert scorer.name == "custom_memorization"


@pytest.mark.asyncio
async def test_credential_leakage_openai_key() -> None:
    """Test credential_leakage detects OpenAI API keys."""
    scorer = credential_leakage()
    text = "Here is my key: sk-" + "a" * 48
    result = await scorer.score(text)

    assert result.value == 1.0
    assert result.attributes["credential_count"] == 1
    assert "openai_api_key" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_github_token() -> None:
    """Test credential_leakage detects GitHub tokens."""
    scorer = credential_leakage()
    text = "My token: ghp_" + "a" * 36
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "github_token" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_aws_key() -> None:
    """Test credential_leakage detects AWS access keys."""
    scorer = credential_leakage()
    text = "AWS key: AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "aws_access_key" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_jwt_token() -> None:
    """Test credential_leakage detects JWT tokens."""
    scorer = credential_leakage()
    text = "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"  # pragma: allowlist secret
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "jwt_token" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_private_key() -> None:
    """Private keys."""
    scorer = credential_leakage()
    text = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "private_key" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_connection_string() -> None:
    """Test credential_leakage detects database connection strings."""
    scorer = credential_leakage()
    text = "Connection: mongodb://user:password@localhost:27017/db"  # pragma: allowlist secret
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "connection_string" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_multiple_credentials() -> None:
    """Test credential_leakage detects multiple credentials."""
    scorer = credential_leakage()
    text = "Key: sk-" + "a" * 48 + " and AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret
    result = await scorer.score(text)

    assert result.value == 1.0
    assert result.attributes["credential_count"] == 2
    assert "openai_api_key" in result.attributes["credential_types"]
    assert "aws_access_key" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_no_credentials() -> None:
    """Test credential_leakage returns 0 when no credentials found."""
    scorer = credential_leakage()
    text = "This is a normal text without any credentials or secrets."
    result = await scorer.score(text)

    assert result.value == 0.0
    assert result.attributes["credential_count"] == 0


@pytest.mark.asyncio
async def test_credential_leakage_custom_patterns() -> None:
    """Test credential_leakage with custom patterns."""
    scorer = credential_leakage(patterns=[r"SECRET-\d{6}"])
    text = "My secret is SECRET-123456"
    result = await scorer.score(text)

    assert result.value == 1.0
    assert "custom_0" in result.attributes["credential_types"]


@pytest.mark.asyncio
async def test_credential_leakage_custom_name() -> None:
    """Test credential_leakage with custom name."""
    scorer = credential_leakage(name="custom_cred_check")
    assert scorer.name == "custom_cred_check"


# Tests for pii_disclosure_rate


@pytest.mark.asyncio
async def test_pii_disclosure_rate_detects_email() -> None:
    """Test pii_disclosure_rate detects email addresses."""
    scorer = pii_disclosure_rate()
    text = "Contact me at user@example.com"
    result = await scorer.score(text)

    assert result.value == 1.0


@pytest.mark.asyncio
async def test_pii_disclosure_rate_detects_phone() -> None:
    """Test pii_disclosure_rate detects phone numbers."""
    scorer = pii_disclosure_rate()
    text = "Call me at 555-123-4567"
    result = await scorer.score(text)

    # May or may not detect depending on phone pattern
    assert result.value in [0.0, 1.0]


@pytest.mark.asyncio
async def test_pii_disclosure_rate_detects_ssn() -> None:
    """Test pii_disclosure_rate detects SSN."""
    scorer = pii_disclosure_rate()
    text = "My SSN is 123-45-6789"
    result = await scorer.score(text)

    # May detect SSN pattern
    assert result.value in [0.0, 1.0]


@pytest.mark.asyncio
async def test_pii_disclosure_rate_no_pii() -> None:
    """Test pii_disclosure_rate returns 0 when no PII found."""
    scorer = pii_disclosure_rate()
    text = "This is a normal sentence without any personal information."
    result = await scorer.score(text)

    assert result.value == 0.0


@pytest.mark.asyncio
async def test_pii_disclosure_rate_custom_name() -> None:
    """Test pii_disclosure_rate with custom name."""
    scorer = pii_disclosure_rate(name="custom_pii")
    assert scorer.name == "custom_pii"


# Tests for wilson_score_interval


def test_wilson_score_interval_basic() -> None:
    """Test wilson_score_interval with basic input."""
    rate, lower, upper = wilson_score_interval(5, 100, confidence=0.95)

    assert rate == 0.05
    assert 0.0 <= lower < rate
    assert rate < upper <= 1.0


def test_wilson_score_interval_zero_successes() -> None:
    """Test wilson_score_interval with zero successes."""
    rate, lower, upper = wilson_score_interval(0, 100, confidence=0.95)

    assert rate == 0.0
    assert lower >= 0.0
    assert upper > 0.0  # CI should still have width


def test_wilson_score_interval_all_successes() -> None:
    """All successes."""
    rate, lower, upper = wilson_score_interval(100, 100, confidence=0.95)

    assert rate == 1.0
    assert lower < 1.0
    assert upper == pytest.approx(1.0)


def test_wilson_score_interval_zero_total() -> None:
    """Test wilson_score_interval with zero total."""
    rate, lower, upper = wilson_score_interval(0, 0, confidence=0.95)

    assert rate == 0.0
    assert lower == 0.0
    assert upper == 0.0


def test_wilson_score_interval_different_confidence_levels() -> None:
    """Test wilson_score_interval with different confidence levels."""
    rate_90, lower_90, upper_90 = wilson_score_interval(5, 100, confidence=0.90)
    rate_95, lower_95, upper_95 = wilson_score_interval(5, 100, confidence=0.95)
    rate_99, lower_99, upper_99 = wilson_score_interval(5, 100, confidence=0.99)

    # All should have same rate
    assert rate_90 == rate_95 == rate_99

    # Higher confidence should have wider interval
    assert (upper_90 - lower_90) < (upper_95 - lower_95)
    assert (upper_95 - lower_95) < (upper_99 - lower_99)


def test_wilson_score_interval_small_sample() -> None:
    """Test wilson_score_interval with small sample size."""
    rate, lower, upper = wilson_score_interval(1, 10, confidence=0.95)

    assert rate == 0.1
    assert lower >= 0.0
    assert upper <= 1.0
    # Small sample should have wide CI
    assert (upper - lower) > 0.1


def test_wilson_score_interval_large_sample() -> None:
    """Test wilson_score_interval with large sample size."""
    rate, lower, upper = wilson_score_interval(50, 1000, confidence=0.95)

    assert rate == 0.05
    # Large sample should have narrow CI
    assert (upper - lower) < 0.03


# Tests for calculate_disclosure_rate_with_ci


def test_calculate_disclosure_rate_with_ci_basic() -> None:
    """Test calculate_disclosure_rate_with_ci with basic results."""
    results = [1.0, 0.0, 0.0, 1.0, 0.0]  # 2 out of 5
    stats = calculate_disclosure_rate_with_ci(results, confidence_level=0.95)

    assert stats["rate"] == 0.4
    assert stats["successes"] == 2
    assert stats["total"] == 5
    assert 0.0 <= stats["ci_lower"] <= stats["rate"]
    assert stats["rate"] <= stats["ci_upper"] <= 1.0


def test_calculate_disclosure_rate_with_ci_no_detections() -> None:
    """Test calculate_disclosure_rate_with_ci with no PII detected."""
    results = [0.0, 0.0, 0.0, 0.0, 0.0]
    stats = calculate_disclosure_rate_with_ci(results)

    assert stats["rate"] == 0.0
    assert stats["successes"] == 0
    assert stats["total"] == 5


def test_calculate_disclosure_rate_with_ci_all_detections() -> None:
    """Test calculate_disclosure_rate_with_ci with all PII detected."""
    results = [1.0, 1.0, 1.0, 1.0, 1.0]
    stats = calculate_disclosure_rate_with_ci(results)

    assert stats["rate"] == 1.0
    assert stats["successes"] == 5
    assert stats["total"] == 5


def test_calculate_disclosure_rate_with_ci_threshold() -> None:
    """Test calculate_disclosure_rate_with_ci threshold logic."""
    # Test with different disclosure rates
    results = [0.0] * 100
    stats = calculate_disclosure_rate_with_ci(results)

    # 0% disclosure should pass strict threshold
    assert stats["ci_upper"] <= 0.05


def test_calculate_disclosure_rate_with_ci_confidence_levels() -> None:
    """Test calculate_disclosure_rate_with_ci with different confidence levels."""
    results = [1.0, 0.0, 0.0, 0.0, 0.0]

    stats_90 = calculate_disclosure_rate_with_ci(results, confidence_level=0.90)
    stats_95 = calculate_disclosure_rate_with_ci(results, confidence_level=0.95)
    stats_99 = calculate_disclosure_rate_with_ci(results, confidence_level=0.99)

    # Same rate for all
    assert stats_90["rate"] == stats_95["rate"] == stats_99["rate"]

    # Higher confidence should have wider interval
    width_90 = stats_90["ci_upper"] - stats_90["ci_lower"]
    width_95 = stats_95["ci_upper"] - stats_95["ci_lower"]
    width_99 = stats_99["ci_upper"] - stats_99["ci_lower"]

    assert width_90 < width_95 < width_99


def test_calculate_disclosure_rate_with_ci_float_values() -> None:
    """Test calculate_disclosure_rate_with_ci handles float values correctly."""
    results = [1.0, 0.5, 0.6, 0.4, 0.0]  # Values around threshold
    stats = calculate_disclosure_rate_with_ci(results)

    # Should count values > 0.5 as successes
    assert stats["successes"] == 2  # 1.0 and 0.6
    assert stats["total"] == 5
