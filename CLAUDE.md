# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- `poetry install` - Install dependencies
- `poetry install --extras multimodal` - Install with media processing (audio, video, images)
- `poetry install --extras training` - Install with ML model integration
- `poetry install --all-extras` - Install all optional features

### Code Quality
- `ruff check` - Run linter
- `ruff format` - Format code
- `mypy .` - Run type checking
- `pytest` - Run tests
- `pytest tests/test_example.py::test_function` - Run specific test

### Pre-commit
- `pre-commit install` - Install pre-commit hooks
- `pre-commit run --all-files` - Run all pre-commit hooks

## Architecture Overview

### Core Components

**Main SDK Class (`dreadnode/main.py`)**
- `Dreadnode` class is the core SDK interface
- `DEFAULT_INSTANCE` provides module-level functions (`dreadnode.configure()`, `dreadnode.run()`, etc.)
- Handles configuration, initialization, and OpenTelemetry setup
- Manages file system abstraction (local or S3) for artifact storage

**Task System (`dreadnode/task.py`)**
- `Task[P, R]` - Decorator-based task wrapper for functions
- Supports both sync and async functions
- Built-in scoring, input/output logging, execution metrics
- Task mapping and batch execution (`map()`, `top_n()`, `try_*` variants)

**Tracing & Spans (`dreadnode/tracing/span.py`)**
- `RunSpan` - Top-level execution context for experiments
- `TaskSpan` - Individual task execution tracking
- Built on OpenTelemetry with custom exporters

**Data Types (`dreadnode/data_types/`)**
- Rich object types: `Text`, `Image`, `Audio`, `Video`, `Object3D`, `Table`
- Automatic serialization and storage
- Conversion utilities in `dreadnode/convert.py`

**API Client (`dreadnode/api/client.py`)**
- REST API client for Dreadnode server
- Data export capabilities (runs, metrics, parameters, timeseries)
- Project and run management

### Key Patterns

**Decorator Usage**
```python
@dreadnode.task(scorers=[score_finding])
@dreadnode.scorer(name="Score Finding") 
def scorer_func(output) -> float: ...
```

**Context Managers**
```python
with dreadnode.run("experiment-name"):
    with dreadnode.task_span("subtask"):
        # work here
```

**Logging Functions**
- `log_param()`, `log_params()` - Configuration/hyperparameters
- `log_metric()`, `log_metrics()` - Measurements with aggregation modes
- `log_input()`, `log_output()` - Runtime objects
- `log_artifact()` - Files/directories

## Configuration

### Environment Variables
- `DREADNODE_SERVER_URL` or `DREADNODE_SERVER` - API server URL
- `DREADNODE_API_TOKEN` or `DREADNODE_API_KEY` - Authentication token
- `DREADNODE_LOCAL_DIR` - Local storage directory
- `DREADNODE_PROJECT` - Default project name

### Initialization
Call `dreadnode.configure()` before using the SDK. The library supports:
- Server + token: Full cloud functionality
- Local directory: Offline file-based storage
- No configuration: In-memory only (with warnings)

## Rigging Integration Patterns

The Dreadnode SDK is designed to work seamlessly with the [Rigging](https://github.com/dreadnode/rigging) library for LLM interactions:

### Task + Rigging Decorator Combo
**Standard Pattern**: Layer `@dreadnode.task` on top of `@rg.prompt` decorators:
```python
import rigging as rg
import dreadnode as dn

@dn.task(scorers=[count_jokes], log_inputs={"topic"})
@rg.prompt(generator_id="gpt-4o-mini")
async def generate_jokes(count: int, topic: str) -> list[Joke]:
    "Write {{count}} short hilarious jokes about {{topic}}."

@dn.task(log_inputs=True)
@rg.prompt(generator_id="gpt-4o", tools=[generate_jokes])
async def write_joke(topic: str) -> rg.Chat:
    """Generate some jokes, then choose the best."""
```

### LLM Agent Evaluation Patterns
```python
@dn.task(name="Find Credentials")
@rg.prompt
def find_creds(directory: str) -> list[Credentials]:
    """Read and enumerate any credentials from files in the provided directory."""

# Tools integration
@rg.prompt(tools=[reversing_tools])
async def analyze_binary(binary: str) -> list[Finding]:
    """Analyze the binary for vulnerabilities."""
```

### Data Logging with Rigging
- Rigging chat objects can be logged as inputs/outputs
- Tool call sequences are automatically tracked
- LLM responses integrate with Dreadnode's rich object storage

## Coding Style Guide

### Type Hints (Critical - Author is Very Particular)

**Full Coverage Required**: Every function, method, class, and variable must have complete type annotations:
```python
def log_metric(
    self,
    name: str,
    value: float | bool | Metric,
    *,
    step: int = 0,
    origin: t.Any | None = None,
    timestamp: datetime | None = None,
    mode: MetricAggMode | None = None,
    attributes: AnyDict | None = None,
    to: ToObject = "task-or-run",
) -> Metric:
```

**Modern Type Syntax**: Always use Python 3.10+ features:
- Union with `|`: `str | None` not `Union[str, None]`
- Import pattern: `import typing as t`
- Explicit `t.Any`, `t.Literal`, etc.
- Generic type parameters: `T = t.TypeVar("T")`

**Complex Type Patterns**:
```python
# Type aliases for readability
AnyDict = dict[str, t.Any]
ToObject = t.Literal["task-or-run", "run"]

# Generic classes
class Task(t.Generic[P, R]):
    func: t.Callable[P, R]

# Overloads for complex method signatures
@t.overload
def task(self, *, scorers: None = None, ...) -> TaskDecorator: ...

@t.overload
def task(self, *, scorers: t.Sequence[Scorer[R]], ...) -> ScoredTaskDecorator[R]: ...
```

### Naming Conventions

**Functions/Variables**: `snake_case` with descriptive names:
- `log_metric`, `push_update`, `seems_useful_to_serialize`
- Boolean functions: `is_default`, `_test_connection`

**Classes**: `PascalCase` with clear semantics:
- `Dreadnode`, `TaskSpan`, `DreadnodeConfigWarning`

**Constants**: `SCREAMING_SNAKE_CASE`:
- `DEFAULT_SERVER_URL`, `ENV_API_TOKEN`, `INHERITED`

### Error Handling Patterns

**Decorator-based Internal Error Handling**:
```python
@handle_internal_errors()
def log_param(self, key: str, value: JsonValue) -> None:
```

**User-facing Warnings**:
```python
warn_at_user_stacklevel(
    "log_params() was called outside of a run.",
    category=DreadnodeUsageWarning,
)
```

**Custom Exception Hierarchies**:
```python
class DreadnodeConfigWarning(UserWarning):
    pass
```

### Code Organization

**Import Order**:
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical) 
3. Local imports (alphabetical)

**Keyword-only Arguments**: Use `*` extensively for clarity:
```python
def task(self, *, scorers=None, name=None, tags=None): ...
```

**Context Managers**: Preferred for resource management and scoping:
```python
with dreadnode.run("experiment"):
    with dreadnode.task_span("subtask"):
        # work
```

### Documentation Style

**Google/Napoleon Docstrings** with examples:
```python
def configure(self, *, server: str | None = None) -> None:
    """
    Configure the Dreadnode SDK.
    
    Args:
        server: The Dreadnode server URL.
        
    Example:
        ```
        dreadnode.configure(server="https://api.dreadnode.io")
        ```
    """
```

## Development Notes

- Uses strict mypy configuration (`strict = true`)
- Ruff linting with comprehensive rule set (line length 100)
- Poetry for dependency management with optional extras
- Pre-commit hooks for code quality
- Async/await support throughout with hybrid sync/async patterns
- OpenTelemetry integration for distributed tracing
- S3-compatible storage with automatic endpoint resolution for Docker environments
- Dataclass-based configuration with environment variable integration
- Production-ready patterns: lifecycle management, observability, graceful error handling