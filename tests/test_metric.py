from datetime import datetime, timezone

from dreadnode.metric import Metric, Scorer

# ruff: noqa: PLR2004


def test_metric_creation() -> None:
    """Test basic Metric creation."""
    # Test with minimal arguments
    metric = Metric(value=0.5)
    assert metric.value == 0.5
    assert metric.step == 0
    assert isinstance(metric.timestamp, datetime)
    assert metric.attributes == {}

    # Test with all arguments
    timestamp = datetime.now(timezone.utc)
    metric = Metric(value=0.75, step=1, timestamp=timestamp, attributes={"source": "test"})
    assert metric.value == 0.75
    assert metric.step == 1
    assert metric.timestamp == timestamp
    assert metric.attributes == {"source": "test"}


def test_metric_from_many() -> None:
    """Test creating a composite metric from individual values and weights."""
    values = [
        ("metric1", 0.8, 1.0),  # name, value, weight
        ("metric2", 0.6, 0.5),  # name, value, weight
    ]

    # Expected weighted average: (0.8*1.0 + 0.6*0.5) / (1.0 + 0.5) = 0.8+0.3 / 1.5 = 0.733
    metric = Metric.from_many(values, step=1, source="test")

    # Check the weighted average
    assert round(metric.value, 3) == 0.733
    assert metric.step == 1
    assert metric.attributes["source"] == "test"

    # Check that individual metrics are stored in attributes
    assert "metric1" in metric.attributes
    assert "metric2" in metric.attributes
    assert metric.attributes["metric1"] == 0.8
    assert metric.attributes["metric2"] == 0.6


def test_scorer() -> None:
    """Test Scorer functionality."""

    # Create a simple scorer function
    def accuracy_scorer(prediction, target):
        return 1.0 if prediction == target else 0.0

    # Create a scorer
    scorer = Scorer.from_callable(accuracy_scorer, name="accuracy", higher_is_better=True)

    # Check scorer properties
    assert scorer.name == "accuracy"
    assert scorer.higher_is_better is True

    # Test scoring
    result = scorer("cat", "cat")
    assert result.value == 1.0
    assert result.attributes["name"] == "accuracy"

    result = scorer("cat", "dog")
    assert result.value == 0.0
    assert result.attributes["name"] == "accuracy"


def test_log_metric(configured_instance) -> None:
    """Test logging metrics through the dreadnode instance."""
    with configured_instance.run(name="test-run"):
        # Log a simple metric
        configured_instance.log_metric("accuracy", 0.8)

        # Test with attributes
        configured_instance.log_metric("precision", 0.75, source="test")

        # Get the metrics for the current run
        metrics_dict = configured_instance.get_metrics()

        # Check that metrics were logged
        assert "accuracy" in metrics_dict
        assert "precision" in metrics_dict
        assert metrics_dict.get("accuracy").value == 0.8
        assert metrics_dict.get("precision").value == 0.75
        assert metrics_dict.get("precision").attributes["source"] == "test"


def test_metric_aggregation(configured_instance) -> None:
    """Test metric aggregation modes."""
    with configured_instance.run(name="test-run"):
        # Log multiple metrics with same name
        configured_instance.log_metric("score", 1.0, step=1)
        configured_instance.log_metric("score", 3.0, step=2)
        configured_instance.log_metric("score", 2.0, step=3)

        metrics_dict = configured_instance.get_metrics()

        # Test various aggregation modes
        avg = metrics_dict.aggregate("score", mode="avg")
        assert avg == 2.0  # (1.0 + 3.0 + 2.0) / 3

        sum_val = metrics_dict.aggregate("score", mode="sum")
        assert sum_val == 6.0  # 1.0 + 3.0 + 2.0

        min_val = metrics_dict.aggregate("score", mode="min")
        assert min_val == 1.0

        max_val = metrics_dict.aggregate("score", mode="max")
        assert max_val == 3.0

        count = metrics_dict.aggregate("score", mode="count")
        assert count == 3
