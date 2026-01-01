"""Comprehensive tests for core/scorer.py - Scorer class and composition functions."""

import pytest
from copy import deepcopy

from dreadnode.core.scorer import (
    Scorer,
    ScorerWarning,
    scorer,
    # Composition functions
    invert,
    remap_range,
    normalize,
    threshold,
    and_,
    or_,
    not_,
    add,
    subtract,
    avg,
    weighted_avg,
    scale,
    clip,
    equals,
    forward,
)
from dreadnode.core.metric import Metric
from dreadnode.core.exceptions import AssertionFailedError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_scorer():
    """A simple scorer that returns the input value as a float."""

    @Scorer
    async def identity(x: float) -> float:
        return x

    return identity


@pytest.fixture
def length_scorer():
    """A scorer that returns string length / 100."""

    @Scorer
    async def length(text: str) -> float:
        return len(text) / 100.0

    return length


@pytest.fixture
def always_one_scorer():
    """A scorer that always returns 1.0."""

    @Scorer
    async def always_one(x) -> float:
        return 1.0

    return always_one


@pytest.fixture
def always_zero_scorer():
    """A scorer that always returns 0.0."""

    @Scorer
    async def always_zero(x) -> float:
        return 0.0

    return always_zero


@pytest.fixture
def error_scorer():
    """A scorer that raises an exception."""

    @Scorer
    async def error(x) -> float:
        raise ValueError("Test error")

    return error


@pytest.fixture
def sync_scorer():
    """A synchronous scorer function."""

    def sync_score(x: float) -> float:
        return x * 2

    return Scorer(sync_score, name="sync_scorer")


@pytest.fixture
def metric_returning_scorer():
    """A scorer that returns a Metric directly."""

    @Scorer
    async def metric_scorer(x: float) -> Metric:
        return Metric(value=x, step=5, attributes={"custom": "value"})

    return metric_scorer


@pytest.fixture
def multi_metric_scorer():
    """A scorer that returns multiple metrics."""

    @Scorer
    async def multi(x: float) -> list[float]:
        return [x, x * 2, x * 3]

    return multi


# =============================================================================
# Scorer Class - Basic Creation and Properties
# =============================================================================


class TestScorerCreation:
    """Tests for Scorer instantiation and basic properties."""

    def test_create_from_async_function(self):
        """Test creating a Scorer from an async function."""

        async def score_func(x: float) -> float:
            return x

        scorer = Scorer(score_func)
        assert scorer.name == "score_func"
        assert scorer.catch is False
        assert scorer.step == 0
        assert scorer.auto_increment_step is False
        assert scorer.log_all is True

    def test_create_from_sync_function(self):
        """Test creating a Scorer from a sync function."""

        def sync_func(x: float) -> float:
            return x

        scorer = Scorer(sync_func, name="my_sync_scorer")
        assert scorer.name == "my_sync_scorer"

    def test_create_with_all_options(self):
        """Test creating a Scorer with all options specified."""

        async def func(x: float) -> float:
            return x

        scorer = Scorer(
            func,
            name="custom_name",
            assert_=True,
            attributes={"key": "value"},
            catch=True,
            step=10,
            auto_increment_step=True,
            log_all=False,
        )

        assert scorer.name == "custom_name"
        assert scorer.assert_ is True
        assert scorer.attributes == {"key": "value"}
        assert scorer.catch is True
        assert scorer.step == 10
        assert scorer.auto_increment_step is True
        assert scorer.log_all is False

    def test_create_from_existing_scorer(self):
        """Test creating a Scorer from another Scorer unwraps the function."""

        async def func(x: float) -> float:
            return x

        original = Scorer(func, name="original")
        wrapped = Scorer(original, name="wrapped")

        assert wrapped.name == "wrapped"
        assert wrapped.func is original.func

    def test_repr(self, simple_scorer):
        """Test string representation."""
        repr_str = repr(simple_scorer)
        assert "Scorer" in repr_str
        assert "identity" in repr_str

    def test_scorer_decorator_direct(self):
        """Test @scorer decorator applied directly."""

        @scorer
        async def my_scorer(x: float) -> float:
            return x

        assert isinstance(my_scorer, Scorer)
        assert my_scorer.name == "my_scorer"

    def test_scorer_decorator_with_options(self):
        """Test @scorer decorator with options."""

        @scorer(name="custom", assert_=True)
        async def my_scorer(x: float) -> float:
            return x

        assert my_scorer.name == "custom"
        assert my_scorer.assert_ is True

    def test_scorer_decorator_on_existing_scorer(self, simple_scorer):
        """Test that @scorer on a Scorer returns it unchanged."""
        result = scorer(simple_scorer)
        assert result is simple_scorer


# =============================================================================
# Scorer Class - fit and fit_many
# =============================================================================


class TestScorerFit:
    """Tests for Scorer.fit and Scorer.fit_many class methods."""

    def test_fit_with_scorer(self, simple_scorer):
        """Test fit() with an existing Scorer returns it."""
        result = Scorer.fit(simple_scorer)
        assert result is simple_scorer

    def test_fit_with_callable(self):
        """Test fit() with a callable wraps it in a Scorer."""

        async def func(x: float) -> float:
            return x

        result = Scorer.fit(func)
        assert isinstance(result, Scorer)
        assert result.func is func

    def test_fit_many_with_none(self):
        """Test fit_many() with None returns empty list."""
        result = Scorer.fit_many(None)
        assert result == []

    def test_fit_many_with_sequence(self, simple_scorer):
        """Test fit_many() with a sequence."""

        async def func(x: float) -> float:
            return x

        result = Scorer.fit_many([simple_scorer, func])
        assert len(result) == 2
        assert result[0] is simple_scorer
        assert isinstance(result[1], Scorer)

    def test_fit_many_with_mapping(self):
        """Test fit_many() with a mapping."""

        async def func1(x: float) -> float:
            return x

        async def func2(x: float) -> float:
            return x * 2

        result = Scorer.fit_many({"scorer1": func1, "scorer2": func2})
        assert len(result) == 2
        names = {s.name for s in result}
        assert names == {"scorer1", "scorer2"}


# =============================================================================
# Scorer Class - Clone and With
# =============================================================================


class TestScorerCloneAndWith:
    """Tests for Scorer cloning and modification methods."""

    def test_clone(self, simple_scorer):
        """Test clone() creates an independent copy."""
        cloned = simple_scorer.clone()
        assert cloned is not simple_scorer
        assert cloned.name == simple_scorer.name
        assert cloned.func is simple_scorer.func

    def test_deepcopy(self, simple_scorer):
        """Test __deepcopy__() works correctly."""
        copied = deepcopy(simple_scorer)
        assert copied is not simple_scorer
        assert copied.name == simple_scorer.name

    def test_with_name(self, simple_scorer):
        """Test with_() can change name."""
        new_scorer = simple_scorer.with_(name="new_name")
        assert new_scorer.name == "new_name"
        assert simple_scorer.name == "identity"  # Original unchanged

    def test_with_attributes(self, simple_scorer):
        """Test with_() can add attributes."""
        new_scorer = simple_scorer.with_(attributes={"key": "value"})
        assert new_scorer.attributes == {"key": "value"}

    def test_with_step(self, simple_scorer):
        """Test with_() can change step."""
        new_scorer = simple_scorer.with_(step=5)
        assert new_scorer.step == 5
        assert simple_scorer.step == 0

    def test_with_catch(self, simple_scorer):
        """Test with_() can change catch."""
        new_scorer = simple_scorer.with_(catch=True)
        assert new_scorer.catch is True

    def test_with_log_all(self, simple_scorer):
        """Test with_() can change log_all."""
        new_scorer = simple_scorer.with_(log_all=False)
        assert new_scorer.log_all is False

    def test_with_auto_increment_step(self, simple_scorer):
        """Test with_() can change auto_increment_step."""
        new_scorer = simple_scorer.with_(auto_increment_step=True)
        assert new_scorer.auto_increment_step is True

    def test_assert_on(self, simple_scorer):
        """Test assert_on() sets assert_ to True."""
        new_scorer = simple_scorer.assert_on()
        assert new_scorer.assert_ is True

    def test_assert_off(self, simple_scorer):
        """Test assert_off() sets assert_ to False."""
        scorer_with_assert = simple_scorer.with_(assert_=True)
        new_scorer = scorer_with_assert.assert_off()
        assert new_scorer.assert_ is False

    def test_rename(self, simple_scorer):
        """Test rename() changes the name."""
        new_scorer = simple_scorer.rename("renamed")
        assert new_scorer.name == "renamed"


# =============================================================================
# Scorer Class - Execution
# =============================================================================


class TestScorerExecution:
    """Tests for Scorer execution methods."""

    @pytest.mark.asyncio
    async def test_score_returns_metric(self, simple_scorer):
        """Test score() returns a Metric."""
        result = await simple_scorer.score(0.5)
        assert isinstance(result, Metric)
        assert result.value == 0.5

    @pytest.mark.asyncio
    async def test_call_returns_metric(self, simple_scorer):
        """Test __call__() returns a Metric."""
        result = await simple_scorer(0.75)
        assert isinstance(result, Metric)
        assert result.value == 0.75

    @pytest.mark.asyncio
    async def test_score_with_sync_function(self, sync_scorer):
        """Test scoring with a synchronous function."""
        result = await sync_scorer.score(2.0)
        assert result.value == 4.0

    @pytest.mark.asyncio
    async def test_normalize_and_score_returns_list(self, simple_scorer):
        """Test normalize_and_score() returns a list of metrics."""
        result = await simple_scorer.normalize_and_score(0.5)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].value == 0.5

    @pytest.mark.asyncio
    async def test_score_composite_returns_tuple(self, simple_scorer):
        """Test score_composite() returns primary and others."""
        primary, others = await simple_scorer.score_composite(0.5)
        assert isinstance(primary, Metric)
        assert primary.value == 0.5
        assert others == []

    @pytest.mark.asyncio
    async def test_metric_returning_scorer(self, metric_returning_scorer):
        """Test scorer that returns a Metric directly."""
        result = await metric_returning_scorer.score(0.5)
        assert result.value == 0.5
        assert result.step == 5
        assert result.attributes.get("custom") == "value"

    @pytest.mark.asyncio
    async def test_multi_metric_scorer(self, multi_metric_scorer):
        """Test scorer that returns multiple metrics."""
        results = await multi_metric_scorer.normalize_and_score(1.0)
        assert len(results) == 3
        assert results[0].value == 1.0
        assert results[1].value == 2.0
        assert results[2].value == 3.0

    @pytest.mark.asyncio
    async def test_auto_increment_step(self):
        """Test auto_increment_step increments after each call."""

        @Scorer
        async def inc_scorer(x: float) -> float:
            return x

        scorer = inc_scorer.with_(auto_increment_step=True, step=0)

        result1 = await scorer.score(1.0)
        assert result1.step == 0
        assert scorer.step == 1

        result2 = await scorer.score(2.0)
        assert result2.step == 1
        assert scorer.step == 2

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Pre-existing recursion bug in warn_at_user_stacklevel")
    async def test_catch_exception(self, error_scorer):
        """Test catch=True catches exceptions."""
        safe_scorer = error_scorer.with_(catch=True)
        result = await safe_scorer.score("test")
        assert result.value == 0.0
        assert "error" in result.attributes

    @pytest.mark.asyncio
    async def test_exception_without_catch(self, error_scorer):
        """Test catch=False raises exceptions."""
        with pytest.raises(ValueError, match="Test error"):
            await error_scorer.score("test")

    @pytest.mark.asyncio
    async def test_log_all_false_returns_single_metric(self, multi_metric_scorer):
        """Test log_all=False returns only primary metric."""
        scorer = multi_metric_scorer.with_(log_all=False)
        results = await scorer.normalize_and_score(1.0)
        assert len(results) == 1


# =============================================================================
# Scorer Class - Binding
# =============================================================================


class TestScorerBinding:
    """Tests for Scorer.bind() method."""

    @pytest.mark.asyncio
    async def test_bind_overrides_input(self, simple_scorer):
        """Test bind() overrides the input object."""
        bound = simple_scorer.bind(0.99)
        result = await bound.score(0.1)  # 0.1 should be ignored
        assert result.value == 0.99

    @pytest.mark.asyncio
    async def test_bind_with_different_types(self, length_scorer):
        """Test bind() with different types."""
        bound = length_scorer.bind("hello world")
        result = await bound.score("ignored")
        assert result.value == pytest.approx(0.11, rel=0.01)


# =============================================================================
# Scorer Class - Adapter (as_scorer)
# =============================================================================


class TestScorerAsScorer:
    """Tests for Scorer.as_scorer() method."""

    @pytest.mark.asyncio
    async def test_as_scorer_adapts_type(self, length_scorer):
        """Test as_scorer() adapts to a different type."""

        # Adapt to score a dict's 'text' field
        adapted = length_scorer.as_scorer(lambda d: d["text"])
        result = await adapted.score({"text": "hello"})
        assert result.value == pytest.approx(0.05, rel=0.01)


# =============================================================================
# Scorer Class - Evaluate (class method)
# =============================================================================


class TestScorerEvaluate:
    """Tests for Scorer.evaluate() class method."""

    @pytest.mark.asyncio
    async def test_evaluate_multiple_scorers(self, simple_scorer, always_one_scorer):
        """Test evaluate() runs multiple scorers."""
        results = await Scorer.evaluate(
            0.5,
            [simple_scorer, always_one_scorer],
        )
        assert "identity" in results
        assert "always_one" in results
        assert results["identity"][0].value == 0.5
        assert results["always_one"][0].value == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_step(self, simple_scorer):
        """Test evaluate() applies step to all metrics."""
        results = await Scorer.evaluate(0.5, [simple_scorer], step=10)
        assert results["identity"][0].step == 10

    @pytest.mark.asyncio
    async def test_evaluate_assert_all_true(self, always_one_scorer, always_zero_scorer):
        """Test evaluate() with assert_scores=True fails on zero."""
        with pytest.raises(AssertionFailedError):
            await Scorer.evaluate(
                "test",
                [always_one_scorer, always_zero_scorer],
                assert_scores=True,
            )

    @pytest.mark.asyncio
    async def test_evaluate_assert_all_false(self, always_zero_scorer):
        """Test evaluate() with assert_scores=False doesn't fail."""
        results = await Scorer.evaluate(
            "test",
            [always_zero_scorer],
            assert_scores=False,
        )
        assert results["always_zero"][0].value == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_assert_specific_scorers(
        self, always_one_scorer, always_zero_scorer
    ):
        """Test evaluate() with specific scorers to assert."""
        # Only assert always_one, not always_zero
        results = await Scorer.evaluate(
            "test",
            [always_one_scorer, always_zero_scorer],
            assert_scores=["always_one"],
        )
        assert "always_one" in results
        assert "always_zero" in results


# =============================================================================
# Scorer Class - Operator Overloads
# =============================================================================


class TestScorerOperators:
    """Tests for Scorer operator overloads."""

    @pytest.mark.asyncio
    async def test_gt_operator(self, simple_scorer):
        """Test > operator creates threshold scorer."""
        threshold_scorer = simple_scorer > 0.5

        result = await threshold_scorer.score(0.6)
        assert result.value == 1.0

        result = await threshold_scorer.score(0.4)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_lt_operator(self, simple_scorer):
        """Test < operator creates threshold scorer."""
        threshold_scorer = simple_scorer < 0.5

        result = await threshold_scorer.score(0.4)
        assert result.value == 1.0

        result = await threshold_scorer.score(0.6)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_ge_operator(self, simple_scorer):
        """Test >= operator creates threshold scorer."""
        threshold_scorer = simple_scorer >= 0.5

        result = await threshold_scorer.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_le_operator(self, simple_scorer):
        """Test <= operator creates threshold scorer."""
        threshold_scorer = simple_scorer <= 0.5

        result = await threshold_scorer.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_and_operator(self, always_one_scorer, always_zero_scorer):
        """Test & operator creates AND scorer."""
        and_scorer = always_one_scorer & always_zero_scorer
        result = await and_scorer.score("test")
        assert result.value == 0.0  # AND fails because one is 0

        and_both = always_one_scorer & always_one_scorer
        result = await and_both.score("test")
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_or_operator(self, always_one_scorer, always_zero_scorer):
        """Test | operator creates OR scorer."""
        or_scorer = always_one_scorer | always_zero_scorer
        result = await or_scorer.score("test")
        assert result.value == 1.0  # OR passes because one is 1

        or_both_zero = always_zero_scorer | always_zero_scorer
        result = await or_both_zero.score("test")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_invert_operator(self, always_one_scorer, always_zero_scorer):
        """Test ~ operator creates NOT scorer."""
        not_one = ~always_one_scorer
        result = await not_one.score("test")
        assert result.value == 0.0  # NOT of truthy is falsy

        not_zero = ~always_zero_scorer
        result = await not_zero.score("test")
        assert result.value == 1.0  # NOT of falsy is truthy

    @pytest.mark.asyncio
    async def test_add_operator(self, simple_scorer):
        """Test + operator creates add scorer."""

        @Scorer
        async def double(x: float) -> float:
            return x * 2

        add_scorer = simple_scorer + double
        result = await add_scorer.score(1.0)
        assert result.value == 3.0  # 1.0 + 2.0

    @pytest.mark.asyncio
    async def test_sub_operator(self, simple_scorer):
        """Test - operator creates subtract scorer."""

        @Scorer
        async def half(x: float) -> float:
            return x / 2

        sub_scorer = simple_scorer - half
        result = await sub_scorer.score(1.0)
        assert result.value == 0.5  # 1.0 - 0.5

    @pytest.mark.asyncio
    async def test_mul_operator(self, simple_scorer):
        """Test * operator creates scale scorer."""
        scaled = simple_scorer * 2.0
        result = await scaled.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_rmul_operator(self, simple_scorer):
        """Test reverse * operator creates scale scorer."""
        scaled = 2.0 * simple_scorer
        result = await scaled.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_truediv_operator(self, simple_scorer):
        """Test / operator creates scale scorer with inverse."""
        scaled = simple_scorer / 2.0
        result = await scaled.score(1.0)
        assert result.value == 0.5

    def test_rshift_operator(self, simple_scorer):
        """Test >> operator renames with log_all=True."""
        renamed = simple_scorer >> "new_name"
        assert renamed.name == "new_name"
        assert renamed.log_all is True

    def test_floordiv_operator(self, simple_scorer):
        """Test // operator renames with log_all=False."""
        renamed = simple_scorer // "new_name"
        assert renamed.name == "new_name"
        assert renamed.log_all is False


# =============================================================================
# Composition Functions - Invert
# =============================================================================


class TestInvert:
    """Tests for invert() function."""

    @pytest.mark.asyncio
    async def test_invert_basic(self, simple_scorer):
        """Test basic inversion."""
        inverted = invert(simple_scorer)
        result = await inverted.score(0.3)
        assert result.value == pytest.approx(0.7, rel=0.01)

    @pytest.mark.asyncio
    async def test_invert_with_known_max(self, simple_scorer):
        """Test inversion with custom known_max."""
        inverted = invert(simple_scorer, known_max=10.0)
        result = await inverted.score(3.0)
        assert result.value == pytest.approx(7.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_invert_name(self, simple_scorer):
        """Test inverted scorer has correct name."""
        inverted = invert(simple_scorer)
        assert inverted.name == "identity_inverted"

    @pytest.mark.asyncio
    async def test_invert_custom_name(self, simple_scorer):
        """Test inverted scorer with custom name."""
        inverted = invert(simple_scorer, name="custom_invert")
        assert inverted.name == "custom_invert"


# =============================================================================
# Composition Functions - Remap and Normalize
# =============================================================================


class TestRemapRange:
    """Tests for remap_range() function."""

    @pytest.mark.asyncio
    async def test_remap_basic(self, simple_scorer):
        """Test basic range remapping."""
        remapped = remap_range(
            simple_scorer,
            known_min=0.0,
            known_max=1.0,
            new_min=0.0,
            new_max=100.0,
        )
        result = await remapped.score(0.5)
        assert result.value == pytest.approx(50.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_remap_different_ranges(self, simple_scorer):
        """Test remapping between different ranges."""
        remapped = remap_range(
            simple_scorer,
            known_min=0.0,
            known_max=10.0,
            new_min=100.0,
            new_max=200.0,
        )
        result = await remapped.score(5.0)
        assert result.value == pytest.approx(150.0, rel=0.01)

    def test_remap_invalid_ranges(self, simple_scorer):
        """Test remap_range raises on invalid ranges."""
        with pytest.raises(ValueError):
            remap_range(
                simple_scorer,
                known_min=1.0,
                known_max=0.0,  # Invalid: min > max
                new_min=0.0,
                new_max=1.0,
            )


class TestNormalize:
    """Tests for normalize() function."""

    @pytest.mark.asyncio
    async def test_normalize_basic(self, simple_scorer):
        """Test basic normalization to [0, 1]."""
        normalized = normalize(simple_scorer, known_max=100.0)
        result = await normalized.score(50.0)
        assert result.value == pytest.approx(0.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_normalize_with_min(self, simple_scorer):
        """Test normalization with custom min."""
        normalized = normalize(simple_scorer, known_max=100.0, known_min=50.0)
        result = await normalized.score(75.0)
        assert result.value == pytest.approx(0.5, rel=0.01)


# =============================================================================
# Composition Functions - Threshold
# =============================================================================


class TestThreshold:
    """Tests for threshold() function."""

    @pytest.mark.asyncio
    async def test_threshold_gt(self, simple_scorer):
        """Test greater-than threshold."""
        thresh = threshold(simple_scorer, gt=0.5)

        result = await thresh.score(0.6)
        assert result.value == 1.0

        result = await thresh.score(0.5)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_threshold_gte(self, simple_scorer):
        """Test greater-than-or-equal threshold."""
        thresh = threshold(simple_scorer, gte=0.5)

        result = await thresh.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_threshold_lt(self, simple_scorer):
        """Test less-than threshold."""
        thresh = threshold(simple_scorer, lt=0.5)

        result = await thresh.score(0.4)
        assert result.value == 1.0

        result = await thresh.score(0.5)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_threshold_lte(self, simple_scorer):
        """Test less-than-or-equal threshold."""
        thresh = threshold(simple_scorer, lte=0.5)

        result = await thresh.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_threshold_eq(self, simple_scorer):
        """Test equality threshold."""
        thresh = threshold(simple_scorer, eq=0.5)

        result = await thresh.score(0.5)
        assert result.value == 1.0

        result = await thresh.score(0.51)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_threshold_ne(self, simple_scorer):
        """Test not-equal threshold."""
        thresh = threshold(simple_scorer, ne=0.5)

        result = await thresh.score(0.51)
        assert result.value == 1.0

        result = await thresh.score(0.5)
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_threshold_custom_values(self, simple_scorer):
        """Test threshold with custom pass/fail values."""
        thresh = threshold(simple_scorer, gt=0.5, pass_value=100.0, fail_value=-1.0)

        result = await thresh.score(0.6)
        assert result.value == 100.0

        result = await thresh.score(0.4)
        assert result.value == -1.0


# =============================================================================
# Composition Functions - Logical (and_, or_, not_)
# =============================================================================


class TestLogicalOperations:
    """Tests for logical composition functions."""

    @pytest.mark.asyncio
    async def test_and_both_true(self, always_one_scorer):
        """Test and_() with both scorers truthy."""
        result_scorer = and_(always_one_scorer, always_one_scorer)
        result = await result_scorer.score("test")
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_and_one_false(self, always_one_scorer, always_zero_scorer):
        """Test and_() with one scorer falsy."""
        result_scorer = and_(always_one_scorer, always_zero_scorer)
        result = await result_scorer.score("test")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_or_one_true(self, always_one_scorer, always_zero_scorer):
        """Test or_() with one scorer truthy."""
        result_scorer = or_(always_one_scorer, always_zero_scorer)
        result = await result_scorer.score("test")
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_or_both_false(self, always_zero_scorer):
        """Test or_() with both scorers falsy."""
        result_scorer = or_(always_zero_scorer, always_zero_scorer)
        result = await result_scorer.score("test")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_not_true(self, always_one_scorer):
        """Test not_() with truthy scorer."""
        result_scorer = not_(always_one_scorer)
        result = await result_scorer.score("test")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_not_false(self, always_zero_scorer):
        """Test not_() with falsy scorer."""
        result_scorer = not_(always_zero_scorer)
        result = await result_scorer.score("test")
        assert result.value == 1.0


# =============================================================================
# Composition Functions - Arithmetic (add, subtract, avg, weighted_avg, scale, clip)
# =============================================================================


class TestArithmeticOperations:
    """Tests for arithmetic composition functions."""

    @pytest.mark.asyncio
    async def test_add_two_scorers(self, simple_scorer):
        """Test add() with two scorers."""

        @Scorer
        async def double(x: float) -> float:
            return x * 2

        result_scorer = add(simple_scorer, double)
        result = await result_scorer.score(1.0)
        assert result.value == 3.0  # 1 + 2

    @pytest.mark.asyncio
    async def test_add_with_average(self, simple_scorer):
        """Test add() with average=True."""

        @Scorer
        async def double(x: float) -> float:
            return x * 2

        result_scorer = add(simple_scorer, double, average=True)
        result = await result_scorer.score(1.0)
        assert result.value == 1.5  # (1 + 2) / 2

    def test_add_no_others_raises(self, simple_scorer):
        """Test add() with no other scorers raises."""
        with pytest.raises(ValueError):
            add(simple_scorer)

    @pytest.mark.asyncio
    async def test_subtract(self, simple_scorer):
        """Test subtract() function."""

        @Scorer
        async def half(x: float) -> float:
            return x / 2

        result_scorer = subtract(simple_scorer, half)
        result = await result_scorer.score(1.0)
        assert result.value == 0.5  # 1 - 0.5

    @pytest.mark.asyncio
    async def test_avg(self, simple_scorer):
        """Test avg() function."""

        @Scorer
        async def double(x: float) -> float:
            return x * 2

        result_scorer = avg(simple_scorer, double)
        result = await result_scorer.score(1.0)
        assert result.value == 1.5  # (1 + 2) / 2

    @pytest.mark.asyncio
    async def test_weighted_avg(self):
        """Test weighted_avg() function."""

        @Scorer
        async def one(x) -> float:
            return 1.0

        @Scorer
        async def two(x) -> float:
            return 2.0

        # Weight 1.0 has weight 1, weight 2.0 has weight 2
        # (1*1 + 2*2) / (1+2) = 5/3 = 1.667
        result_scorer = weighted_avg((one, 1.0), (two, 2.0))
        result = await result_scorer.score("test")
        assert result.value == pytest.approx(5.0 / 3.0, rel=0.01)

    def test_weighted_avg_no_scorers_raises(self):
        """Test weighted_avg() with no scorers raises."""
        with pytest.raises(ValueError):
            weighted_avg()

    @pytest.mark.asyncio
    async def test_scale(self, simple_scorer):
        """Test scale() function."""
        scaled = scale(simple_scorer, 2.0)
        result = await scaled.score(0.5)
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_scale_negative(self, simple_scorer):
        """Test scale() with negative factor."""
        scaled = scale(simple_scorer, -1.0)
        result = await scaled.score(0.5)
        assert result.value == -0.5

    @pytest.mark.asyncio
    async def test_clip(self, simple_scorer):
        """Test clip() function."""
        clipped = clip(simple_scorer, 0.3, 0.7)

        result = await clipped.score(0.5)
        assert result.value == 0.5

        result = await clipped.score(0.1)
        assert result.value == 0.3

        result = await clipped.score(0.9)
        assert result.value == 0.7


# =============================================================================
# Composition Functions - Utility (equals, forward)
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility composition functions."""

    @pytest.mark.asyncio
    async def test_equals_match(self):
        """Test equals() when values match."""
        eq_scorer = equals("expected")
        result = await eq_scorer.score("expected")
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_equals_no_match(self):
        """Test equals() when values don't match."""
        eq_scorer = equals("expected")
        result = await eq_scorer.score("different")
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_forward(self):
        """Test forward() returns fixed value."""
        fwd_scorer = forward(0.75)
        result = await fwd_scorer.score("anything")
        assert result.value == 0.75


# =============================================================================
# Complex Composition Scenarios
# =============================================================================


class TestComplexComposition:
    """Tests for complex scorer compositions."""

    @pytest.mark.asyncio
    async def test_chained_operations(self, simple_scorer):
        """Test chaining multiple operations."""
        # (scorer * 2) > 0.5
        chained = (simple_scorer * 2) > 0.5

        result = await chained.score(0.3)
        assert result.value == 1.0  # 0.3 * 2 = 0.6 > 0.5

        result = await chained.score(0.2)
        assert result.value == 0.0  # 0.2 * 2 = 0.4 < 0.5

    @pytest.mark.asyncio
    async def test_nested_logical(self, always_one_scorer, always_zero_scorer):
        """Test nested logical operations."""
        # (one & one) | zero
        nested = (always_one_scorer & always_one_scorer) | always_zero_scorer
        result = await nested.score("test")
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_normalized_then_threshold(self, simple_scorer):
        """Test normalize then threshold."""
        # Normalize 0-100 to 0-1, then check if > 0.5
        composed = normalize(simple_scorer, known_max=100.0)
        composed = composed > 0.5

        result = await composed.score(60.0)
        assert result.value == 1.0

        result = await composed.score(40.0)
        assert result.value == 0.0
