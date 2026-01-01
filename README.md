<p align="center">
    <img
    src="https://d1lppblt9t2x15.cloudfront.net/logos/5714928f3cdc09503751580cffbe8d02.png"
    alt="Logo"
    align="center"
    width="144px"
    height="144px"
    />
</p>

<h3 align="center">
Dreadnode Strikes SDK
</h3>

<h4 align="center">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dreadnode">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/dreadnode">
    <img alt="GitHub License" src="https://img.shields.io/github/license/dreadnode/sdk">
    <img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/dreadnode/sdk/tests.yaml">
    <img alt="Pre-Commit" src="https://img.shields.io/github/actions/workflow/status/dreadnode/sdk/pre-commit.yaml">
    <img alt="Renovate" src="https://img.shields.io/github/actions/workflow/status/dreadnode/sdk/renovate.yaml">
</h4>

</br>

Strikes is a comprehensive platform for building, experimenting with, and evaluating AI security agents.

## Key Features

- **Agents** - Build multi-step reasoning agents with tools, hooks, and scoring
- **Tasks & Runs** - Structure experiments with tracked inputs, outputs, and metrics
- **Evaluations** - Run agents against datasets with composable scorers
- **AIRT** - Adversarial AI Research Tools for security testing (TAP, GOAT, image attacks)
- **Observability** - OpenTelemetry-based tracing with span hierarchy
- **Datasets & Models** - HuggingFace integration with local CAS storage
- **Deployment** - Serve agents via FastAPI, Cloudflare Workers, or Ray

## Quick Example

```python
import dreadnode as dn

dn.configure()

# Define a tool
@dn.tool
def search_database(query: str) -> list[str]:
    """Search the vulnerability database."""
    return ["CVE-2024-1234", "CVE-2024-5678"]

# Create an agent with tools
@dn.agent(model="openai/gpt-4o", tools=[search_database])
def security_analyst():
    """You are a security analyst. Find and analyze vulnerabilities."""

# Run the agent - tracing is automatic
async def main():
    trajectory = await security_analyst.run(
        "Analyze recent vulnerabilities in the database"
    )

    print(f"Completed in {len(trajectory.steps)} steps")
    print(f"Token usage: {trajectory.usage.total_tokens}")
```

## Agents

Create agents with tools, hooks, and real-time scoring:

```python
import dreadnode as dn
from dreadnode import tool
from dreadnode.core.agents.reactions import Finish, Continue

# Tools with type hints
@tool
def scan_ports(host: str) -> list[int]:
    """Scan for open ports on a host."""
    return [22, 80, 443]  # Simplified example

# Agent with configuration
@dn.agent(
    model="anthropic/claude-3-5-sonnet",
    tools=[scan_ports],
    max_steps=10,
)
def pentester():
    """You are a penetration tester. Find security issues."""

# Hooks for control flow
@pentester.hook
async def check_progress(event):
    if "found vulnerability" in str(event):
        return Finish("Vulnerability discovered")
    return Continue()

# Run the agent
trajectory = await pentester.run("Test the web application at localhost:8080")
print(f"Completed in {len(trajectory.steps)} steps")
print(f"Token usage: {trajectory.usage.total_tokens}")
```

## Evaluations

Run systematic evaluations with datasets and scorers:

```python
from dreadnode import Evaluation
from dreadnode.scorers import contains, llm_judge, and_, not_

# Compose scorers
quality = and_(
    contains("vulnerability", case_sensitive=False),
    not_(contains("error")),
)

judge = llm_judge(
    model="openai/gpt-4o-mini",
    rubric="Rate the security analysis from 1-10 based on thoroughness.",
)

# Create evaluation
evaluation = Evaluation(
    name="security-eval",
    task=pentester.as_task(),
    dataset=[
        {"target": "webapp-1", "goal": "Find SQL injection"},
        {"target": "webapp-2", "goal": "Find XSS vulnerabilities"},
        {"target": "api-server", "goal": "Test authentication"},
    ],
    scorers=[quality, judge],
    concurrency=3,
)

# Run evaluation
result = await evaluation.run()
print(f"Average score: {result.metrics['judge'].mean()}")
```

## AIRT (Adversarial AI Research Tools)

Security testing tools for LLMs and classifiers:

```python
from dreadnode.airt import LLMTarget
from dreadnode.airt.attacks import tap_attack, prompt_attack
from dreadnode.airt.transforms.cipher import rot13_cipher, caesar_cipher

# Define target
target = LLMTarget(model="openai/gpt-4o-mini")

# TAP attack (Tree of Attacks)
attack = tap_attack(
    goal="Extract the system prompt",
    target=target,
    attacker_model="openai/gpt-4o-mini",
    evaluator_model="openai/gpt-4o-mini",
    beam_width=10,
)

result = await attack.run(max_iterations=20)
print(f"Best score: {result.best_score}")

# Text transforms for evasion
rot13 = rot13_cipher()
caesar = caesar_cipher(offset=3)
combined = rot13 | caesar  # Compose transforms
```

## Datasets & Models

HuggingFace integration with local storage:

```python
from dreadnode.datasets import Dataset
from dreadnode.models import Model

# Load dataset
dataset = Dataset.from_hf("squad", split="train[:100]")

# Transform and filter
dataset = dataset.map(lambda x: {"input": x["question"]})
dataset = dataset.filter(lambda x: len(x["input"]) > 10)

# Save locally
dataset.save("my-dataset")

# Load models
model = Model.from_hf("bert-base-uncased")
```

## Tracing & Observability

Agents have built-in observability. For lower-level task workflows, use explicit tracing:

```python
import dreadnode as dn

# Agents trace automatically
trajectory = await security_analyst.run("Analyze the target")
# All steps, tool calls, and generations are traced

# For custom task workflows, use explicit runs
@dn.task
async def analyze(target: str) -> dict:
    dn.log_input("target", target)
    result = {"status": "complete"}
    dn.log_output("result", result)
    dn.log_metric("quality", 0.95)
    return result

with dn.run(name="custom-analysis"):
    await analyze("webapp")
```

## Deployment

Serve agents as HTTP endpoints:

```python
from dreadnode.core.integrations.serve import Serve, AuthMode

# Configure server
server = (
    Serve()
    .with_auth(AuthMode.API_KEY)
    .add(security_analyst, path="/analyze")
    .add(pentester, path="/pentest")
)

# Run server
server.run(host="0.0.0.0", port=8000)

# Or get FastAPI app for custom configuration
app = server.app()
```

## Installation

Install from PyPI:

```bash
pip install -U dreadnode
```

With optional features:

```bash
# Multimodal support (audio, video, images)
pip install -U "dreadnode[multimodal]"

# Training integration (transformers callbacks)
pip install -U "dreadnode[training]"

# All optional features
pip install -U "dreadnode[all]"
```

From source:

```bash
git clone https://github.com/dreadnode/sdk
cd sdk
uv sync --all-extras
```

## Notebooks

Comprehensive Jupyter notebook tutorials are available in [`notebooks/`](./notebooks/):

| Category | Notebooks |
|----------|-----------|
| Getting Started | 01: SDK Basics, 02: Tasks & Runs |
| Agents | 03-07: Basics, Tools, Hooks, Scoring, Advanced |
| Evaluation | 08-11: Scorers, Evaluations |
| Data | 12-14: Datasets, Models, Data Types |
| Security (AIRT) | 15-17: Targets, Attacks, Transforms |
| Advanced | 18-24: Search, Generators, Data Designer, Deployment, Tracing, Packaging, Training |
| Developers | 25-27: Config System, Context Injection, Custom Components |
| Environments | 28: Docker, Jupyter Kernel, Kubernetes Sandbox |
| Studies | 29: Optimization Studies, Agent Tuning, Search Strategies |

## Documentation

- **[Installation Guide](https://docs.dreadnode.io/strikes/install)** - Setup options
- **[Introduction](https://docs.dreadnode.io/strikes/intro)** - Getting started guide
- **[API Reference](https://docs.dreadnode.io/strikes/api)** - Complete API documentation

## Examples

Check out **[dreadnode/example-agents](https://github.com/dreadnode/example-agents)** for real-world use cases.

## License

See [LICENSE](./LICENSE) for details.
