# Dreadnode Evaluation Framework

A comprehensive framework for evaluating AI tasks against datasets with support for parameterized scenarios, scoring, assertions, and concurrent execution.

## Overview

The `Evaluation` class provides a complete system for:

- Running tasks against datasets with configurable iterations
- Parameterized scenario testing (grid search over configurations)
- Scoring and assertions on task outputs
- Concurrent sample execution with error handling
- Real-time streaming of evaluation events
- Integration with Dreadnode's tracing system

## Quick Start

```python
from dreadnode.core.evaluations import Evaluation
from dreadnode.core.task import Task
from dreadnode.core.scorer import Scorer

# Define a task
@Task
async def classify(text: str) -> str:
    """Classify text sentiment."""
    # Your classification logic
    return "positive"

# Define a scorer
@Scorer
def accuracy(output: str, expected: str) -> float:
    """Check if output matches expected."""
    return 1.0 if output == expected else 0.0

# Create dataset
dataset = [
    {"text": "I love this!", "expected": "positive"},
    {"text": "This is terrible", "expected": "negative"},
]

# Create and run evaluation
eval = Evaluation(
    task=classify,
    dataset=dataset,
    scorers=[accuracy],
    assert_scores=["accuracy"],
)

result = await eval.run()
print(f"Pass rate: {result.pass_rate:.2%}")
```

## Core Concepts

### Evaluation Configuration

| Parameter               | Type                     | Default  | Description                                |
| ----------------------- | ------------------------ | -------- | ------------------------------------------ |
| `task`                  | `Task \| str`            | Required | Task to evaluate (instance or import path) |
| `dataset`               | `list[In] \| list[dict]` | `None`   | Dataset rows                               |
| `dataset_file`          | `FilePath \| str`        | `None`   | Path to JSONL/CSV/JSON/YAML file           |
| `name`                  | `str`                    | Auto     | Evaluation name                            |
| `iterations`            | `int`                    | `1`      | Runs per scenario per sample               |
| `concurrency`           | `int`                    | `1`      | Parallel sample execution                  |
| `parameters`            | `dict[str, list]`        | `None`   | Parameter grid for scenarios               |
| `scorers`               | `ScorersLike`            | `[]`     | Scoring functions                          |
| `assert_scores`         | `list[str] \| True`      | `[]`     | Scores that must be truthy                 |
| `dataset_input_mapping` | `list \| dict`           | `None`   | Map dataset keys to task params            |
| `preprocessor`          | `Callable`               | `None`   | Dataset transformation function            |
| `trace`                 | `bool`                   | `True`   | Enable tracing                             |

### Dataset Sources

```python
# Inline dataset
eval = Evaluation(task=my_task, dataset=[{"input": "a"}, {"input": "b"}])

# From file (JSONL, CSV, JSON, YAML)
eval = Evaluation(task=my_task, dataset_file="data/test.jsonl")

# From producer function
async def load_data():
    return await fetch_test_cases()

eval = Evaluation(task=my_task, dataset=load_data)

# With preprocessing
def filter_valid(dataset):
    return [row for row in dataset if row.get("valid")]

eval = Evaluation(task=my_task, dataset=raw_data, preprocessor=filter_valid)
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Evaluation.run()                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EvalStart Event                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              For each Parameter Combination (Scenario)          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ScenarioStart                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  For each Iteration (1..iterations)                 │  │  │
│  │  │  ┌───────────────────────────────────────────────┐  │  │  │
│  │  │  │  IterationStart                               │  │  │  │
│  │  │  │  For each Dataset Row (concurrent):           │  │  │  │
│  │  │  │    → Run Task → Score → SampleComplete        │  │  │  │
│  │  │  │  IterationEnd                                 │  │  │  │
│  │  │  └───────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ScenarioEnd                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         EvalEnd Event                           │
│              (finished | max_errors | max_consecutive_errors)   │
└─────────────────────────────────────────────────────────────────┘
```

## Parameterized Scenarios

Run evaluations across parameter combinations (grid search):

```python
eval = Evaluation(
    task=my_task,
    dataset=test_data,
    parameters={
        "temperature": [0.0, 0.5, 1.0],
        "model": ["gpt-4", "claude-3"],
    },
    iterations=3,  # Run each combination 3 times
)

# This creates 6 scenarios (3 temps × 2 models)
# Each scenario runs 3 iterations
# Total runs = 6 × 3 × len(dataset)
```

## Scoring System

### Defining Scorers

```python
from dreadnode.core.scorer import Scorer

# Simple scorer
@Scorer
def exact_match(output: str, expected: str) -> float:
    return 1.0 if output == expected else 0.0

# Scorer with default values
@Scorer
def contains_keyword(output: str, keyword: str = "success") -> float:
    return 1.0 if keyword in output.lower() else 0.0

# Scorer referencing dataset fields
from dreadnode.core.meta import DatasetField

@Scorer
def matches_label(output: str, label: str = DatasetField("ground_truth")) -> float:
    return 1.0 if output == label else 0.0
```

### Assertions

```python
# Assert specific scores must be truthy (> 0)
eval = Evaluation(
    task=my_task,
    dataset=data,
    scorers=[accuracy, relevance, safety],
    assert_scores=["accuracy", "safety"],  # These must pass
)

# Assert ALL scores must be truthy
eval = Evaluation(
    task=my_task,
    dataset=data,
    scorers=[accuracy, relevance],
    assert_scores=True,  # All scores must pass
)
```

## Event System

### Event Types

| Event            | Description                         |
| ---------------- | ----------------------------------- |
| `EvalStart`      | Evaluation begins                   |
| `EvalEnd`        | Evaluation completes                |
| `ScenarioStart`  | Parameter combination begins        |
| `ScenarioEnd`    | Parameter combination completes     |
| `IterationStart` | Iteration within scenario begins    |
| `IterationEnd`   | Iteration within scenario completes |
| `SampleComplete` | Single dataset row processed        |

### Streaming Events

```python
async with eval.stream() as event_stream:
    async for event in event_stream:
        match event:
            case EvalStart():
                print(f"Starting: {event.total_samples} samples")
            case SampleComplete():
                status = "✓" if event.sample.passed else "✗"
                print(f"{status} Sample {event.sample.index}")
            case ScenarioEnd():
                print(f"Scenario pass rate: {event.result.pass_rate:.2%}")
            case EvalEnd():
                print(f"Final: {event.result.passed_count}/{event.result.sample_count}")
```

## Dataset Input Mapping

Control how dataset fields map to task parameters:

```python
# Auto-mapping (default): matches dataset keys to task parameter names
@Task
async def process(text: str, max_length: int) -> str:
    ...

dataset = [{"text": "hello", "max_length": 100}]  # Auto-matched

# Explicit list: only pass specified keys
eval = Evaluation(
    task=process,
    dataset=dataset,
    dataset_input_mapping=["text"],  # Only pass 'text'
)

# Explicit dict: rename keys
dataset = [{"input_text": "hello", "limit": 100}]
eval = Evaluation(
    task=process,
    dataset=dataset,
    dataset_input_mapping={"input_text": "text", "limit": "max_length"},
)
```

## Results Structure

### EvalResult

```python
result = await eval.run()

# Top-level metrics
result.pass_rate        # float: Overall pass rate
result.passed_count     # int: Total passed samples
result.failed_count     # int: Total failed samples
result.error_count      # int: Total errored samples
result.sample_count     # int: Total samples
result.stop_reason      # str: "finished" | "max_errors" | "max_consecutive_errors"

# Nested structure
result.scenarios        # list[ScenarioResult]
```

### ScenarioResult

```python
for scenario in result.scenarios:
    scenario.params         # dict: Parameter values for this scenario
    scenario.pass_rate      # float: Pass rate for scenario
    scenario.iterations     # list[IterationResult]
```

### IterationResult

```python
for iteration in scenario.iterations:
    iteration.iteration     # int: Iteration number
    iteration.samples       # list[Sample]
```

### Sample

```python
for sample in iteration.samples:
    sample.input            # In: Task input
    sample.output           # Out | None: Task output
    sample.scores           # dict[str, float]: Scorer results
    sample.passed           # bool: All assertions passed
    sample.failed           # bool: Any assertion failed or error
    sample.error            # Exception | None: Error if occurred
    sample.index            # int: Position in dataset
    sample.context          # dict: Extra dataset fields
```

## Error Handling

### Error Thresholds

```python
eval = Evaluation(
    task=my_task,
    dataset=large_dataset,
    max_errors=10,              # Stop after 10 total errors
    max_consecutive_errors=3,   # Stop after 3 consecutive errors
)
```

### Stop Reasons

| Reason                     | Description                          |
| -------------------------- | ------------------------------------ |
| `"finished"`               | All samples completed                |
| `"max_errors"`             | Total error threshold exceeded       |
| `"max_consecutive_errors"` | Consecutive error threshold exceeded |

## Console Mode

Run with live terminal display:

```python
result = await eval.console()
```

This provides a rich terminal UI showing:

- Progress bars for scenarios and samples
- Live pass/fail counts
- Error summaries
- Final statistics

## Concurrent Execution

```python
eval = Evaluation(
    task=my_task,
    dataset=large_dataset,
    concurrency=10,  # Process 10 samples in parallel
)
```

Samples within each iteration run concurrently up to the concurrency limit.

## Tracing Integration

```python
# Tracing enabled (default)
eval = Evaluation(task=my_task, dataset=data, trace=True)

# Tracing disabled
eval = Evaluation(task=my_task, dataset=data, trace=False)
```

When enabled, each scenario creates a traced run with:

- Input parameters and dataset configuration
- Per-sample metrics and outputs
- Aggregate statistics (pass_rate, counts)

## Cloning and Modification

```python
# Create modified evaluation
modified = eval.with_(
    iterations=5,
    concurrency=20,
    scorers=[new_scorer],
    append=True,  # Append to existing scorers
)
```

## Type Safety

Use generics for type-safe evaluations:

```python
from dreadnode.core.evaluations import Evaluation

# Typed evaluation
eval: Evaluation[str, dict] = Evaluation(
    task=my_task,  # Task[[str], dict]
    dataset=["input1", "input2"],
)

# Result is typed
result: EvalResult[str, dict] = await eval.run()
```

## Advanced Usage

### Context Variables

Access current dataset row within task execution:

```python
from dreadnode.core.evaluations import current_dataset_row

@Task
async def my_task(text: str) -> str:
    row = current_dataset_row.get()
    if row:
        # Access additional dataset fields
        metadata = row.get("metadata", {})
    return process(text)
```

### Custom Preprocessors

```python
def augment_dataset(dataset):
    """Add computed fields to each row."""
    for row in dataset:
        row["word_count"] = len(row["text"].split())
    return dataset

eval = Evaluation(
    task=my_task,
    dataset=raw_data,
    preprocessor=augment_dataset,
)
```

### Task Discovery

```python
# Reference task by import path
eval = Evaluation(
    task="mypackage.tasks.classify",
    dataset=data,
)
```

## API Reference

### Evaluation Methods

| Method       | Description                          |
| ------------ | ------------------------------------ |
| `run()`      | Execute evaluation and return result |
| `stream()`   | Stream evaluation events             |
| `console()`  | Run with live terminal display       |
| `with_(...)` | Create modified clone                |

### Properties

| Property    | Type  | Description                      |
| ----------- | ----- | -------------------------------- |
| `name`      | `str` | Evaluation name (computed)       |
| `task_name` | `str` | Name of the task being evaluated |

## License

See LICENSE file for details.
