# Dreadnode Agent Framework

A powerful, flexible agent abstraction for building LLM-powered autonomous agents with tool use, event-driven architecture, and comprehensive execution tracing.

## Overview

The `Agent` class provides a complete framework for creating AI agents that can:

- Execute multi-step reasoning with configurable step limits
- Use tools via multiple calling conventions (API, XML, JSON, Pythonic)
- Stream execution events in real-time
- React to events through a hook system
- Track conversation history via trajectories
- Support custom stop conditions

## Quick Start

```python
from dreadnode.core.agents import Agent
from dreadnode.core.tools import tool

# Define a simple tool
@Tool
async def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# Create an agent
agent = Agent(
    name="research-assistant",
    model="anthropic/claude-3-sonnet",
    instructions="You are a helpful research assistant.",
    tools=[search],
    max_steps=5
)

# Run the agent
trajectory = await agent.run("Find information about quantum computing")
```

## Core Concepts

### Agent Configuration

| Parameter         | Type                    | Default  | Description                                |
| ----------------- | ----------------------- | -------- | ------------------------------------------ |
| `model`           | `str \| Generator`      | `None`   | LLM model identifier or Generator instance |
| `instructions`    | `str`                   | `None`   | System instructions for the agent          |
| `max_steps`       | `int`                   | `10`     | Maximum generation + tool call cycles      |
| `tools`           | `list[Tool \| Toolset]` | `[]`     | Available tools for the agent              |
| `tool_mode`       | `ToolMode`              | `"auto"` | Tool calling convention                    |
| `stop_conditions` | `list[StopCondition]`   | `[]`     | Conditions that end execution              |
| `hooks`           | `list[Hook]`            | `[]`     | Event reaction handlers                    |
| `cache`           | `CacheMode`             | `None`   | Message caching strategy                   |

### Tool Modes

The agent supports multiple tool calling conventions:

- **`auto`**: Automatically selects based on model capabilities
- **`api`**: Native function calling (OpenAI/Anthropic style)
- **`xml`**: XML-formatted tool definitions and calls
- **`json-in-xml`**: JSON tool calls wrapped in XML
- **`json-with-tag`**: JSON with explicit tool tags
- **`json`**: Pure JSON tool format
- **`pythonic`**: Python function call syntax

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent.run(input)                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AgentStart Event                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Step Loop (1..max_steps)                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Generate LLM Response → GenerationStep            │  │
│  │  2. Check Stop Conditions                             │  │
│  │  3. Process Tool Calls → ToolStart/ToolEnd/ToolStep   │  │
│  │  4. Handle Reactions from Hooks                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       AgentEnd Event                        │
│         (finished | error | max_steps_reached | stalled)    │
└─────────────────────────────────────────────────────────────┘
```

## Event System

### Event Types

| Event            | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `AgentStart`     | Emitted when agent execution begins                  |
| `AgentEnd`       | Emitted when execution completes                     |
| `AgentError`     | Emitted on generation errors                         |
| `AgentStalled`   | Emitted when no tool calls and stop conditions unmet |
| `GenerationStep` | Emitted after each LLM generation                    |
| `ToolStart`      | Emitted before tool execution                        |
| `ToolEnd`        | Emitted after successful tool execution              |
| `ToolStep`       | Emitted with tool result message                     |
| `ToolError`      | Emitted on tool execution failure                    |
| `ReactStep`      | Emitted when a hook triggers a reaction              |

### Streaming Events

```python
async with agent.stream("Your query here") as event_stream:
    async for event in event_stream:
        match event:
            case GenerationStep():
                print(f"Step {event.step}: Generated response")
            case ToolStart():
                print(f"Calling tool: {event.tool_call.name}")
            case AgentEnd():
                print(f"Finished: {event.stop_reason}")
```

## Hook System

Hooks allow reactive control over agent execution:

```python
from dreadnode.core.agents.reactions import Finish, Retry, Continue

def my_hook(event: AgentEvent) -> Reaction | None:
    if isinstance(event, GenerationStep):
        if "error" in event.messages[-1].content.lower():
            return Retry(feedback="Please try a different approach")
    return None

agent = Agent(
    model="anthropic/claude-3-sonnet",
    hooks=[my_hook]
)
```

### Reaction Types

| Reaction            | Effect                                   |
| ------------------- | ---------------------------------------- |
| `Continue`          | Continue with optional modified messages |
| `Retry`             | Retry the current step                   |
| `RetryWithFeedback` | Retry with additional user feedback      |
| `Finish`            | End execution immediately                |
| `Fail`              | End execution with an error              |

**Priority Order**: `Finish` > `Retry`/`RetryWithFeedback` > `Continue`

## Trajectory & State

The `Trajectory` object tracks the complete execution history:

```python
trajectory = await agent.run("Query")

# Access execution data
print(f"Total tokens: {trajectory.usage.total_tokens}")
print(f"Steps taken: {len(trajectory.steps)}")
print(f"Messages: {trajectory.messages}")
```

### State Management

```python
# Reset agent state
previous_trajectory = agent.reset()

# Clone with modifications
modified_agent = agent.with_(
    max_steps=20,
    tools=[additional_tool],
    append=True  # Append to existing tools
)
```

## Tools

### Defining Tools

```python
from dreadnode.core.tools import tool, Toolset, tool_method

# Simple tool
@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Toolset for grouped tools
class MathTools(Toolset):
    @tool_method
    async def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool_method
    async def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b
```

### Tool Discovery

Tools are automatically discovered from:

- `Tool` instances
- `Toolset` instances
- Objects with `@Tool` decorated methods
- Callable `Component` instances

## Stop Conditions

Define custom conditions for successful completion:

```python
from dreadnode.core.stopping import StopCondition

def has_final_answer(steps: list[AgentStep]) -> bool:
    """Stop when the agent provides a final answer."""
    for step in reversed(steps):
        if isinstance(step, GenerationStep):
            return "FINAL ANSWER:" in step.messages[-1].content
    return False

agent = Agent(
    model="anthropic/claude-3-sonnet",
    stop_conditions=[StopCondition(has_final_answer, name="final_answer")]
)
```

## Advanced Usage

### Custom Generator

```python
from dreadnode.core.generators import Generator

custom_generator = Generator(
    model="custom/model",
    params=GenerateParams(temperature=0.7, max_tokens=2000)
)

agent = Agent(model=custom_generator, ...)
```

### Context Manager Tools

Tools that require setup/teardown can implement async context manager:

```python
class DatabaseTools(Toolset):
    async def __aenter__(self):
        self.connection = await create_connection()
        return self

    async def __aexit__(self, *args):
        await self.connection.close()

    @tool_method
    async def query(self, sql: str) -> str:
        return await self.connection.execute(sql)
```

### Caching

```python
from dreadnode.core.generators.caching import CacheMode

agent = Agent(
    model="anthropic/claude-3-sonnet",
    cache=CacheMode.ENABLED  # Enable message caching
)
```

## Error Handling

The agent handles errors gracefully:

- **Generation errors**: Emit `AgentError`, set `AgentEnd.stop_reason = "error"`
- **Tool errors**: Emit `ToolError`, continue or fail based on configuration
- **Max steps reached**: Emit `AgentEnd.stop_reason = "max_steps_reached"`
- **Stalled execution**: Emit `AgentStalled`, then `AgentEnd.stop_reason = "stalled"`

```python
trajectory = await agent.run("Query")

if trajectory.steps[-1].stop_reason == "error":
    print(f"Error: {trajectory.steps[-1].error}")
```

## Tracing & Observability

The agent integrates with Dreadnode's tracing system:

```python
from dreadnode import log_output

# Automatic token logging
# - input_tokens
# - output_tokens
# - total_tokens

# Custom logging in hooks
def logging_hook(event: AgentEvent):
    if isinstance(event, ToolEnd):
        log_output("tool_result", event.result)
```

## API Reference

### Agent Methods

| Method          | Description                                |
| --------------- | ------------------------------------------ |
| `run(input)`    | Execute agent and return trajectory        |
| `stream(input)` | Stream execution events                    |
| `reset()`       | Clear state and return previous trajectory |
| `with_(...)`    | Create modified clone                      |
| `get_prompt()`  | Generate system prompt                     |

### Properties

| Property     | Type          | Description             |
| ------------ | ------------- | ----------------------- |
| `generator`  | `Generator`   | Resolved LLM generator  |
| `model_name` | `str \| None` | Model identifier        |
| `all_tools`  | `list[Tool]`  | Flattened tool list     |
| `trajectory` | `Trajectory`  | Current execution state |

## License

See LICENSE file for details.
