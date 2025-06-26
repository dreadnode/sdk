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

Strikes is a platform for building, experimenting with, and evaluating AI security agent code.

- **Experiment + Tasking + Observability** in a single place that's lightweight and scales.
- **Track your data** with parameters, inputs, and outputs all connected to your tasks.
- **Log your artifacts** — data, models, files, and folders — to track data of your Dreadnode runs, enabling easy reuse and reproducibility.
- **Measure everything** with metrics throughout your code and anywhere you need them.
- **Scale your code** from a single run to thousands.

```python
import dreadnode as dn
import rigging as rg

from .tools import reversing_tools

dn.configure()

@dataclass
class Finding:
    name: str
    severity: str
    description: str
    exploit_code: str

@dn.scorer(name="Score Finding")
async def score_finding(finding: Finding) -> float:
    if finding.severity == "critical":
        return 1.0
    elif finding.severity == "high":
        return 0.8
    else:
        return 0.2

@dn.task(scorers=[score_finding])
@rg.prompt(tools=[reversing_tools])
async def analyze_binary(binary: str) -> list[Finding]:
    """
    Analyze the binary for vulnerabilities.
    """
    ...

with dn.run(tags=["reverse-engineering"]):
    binary = "c2/downloads/service.exe"

    dn.log_params(
        model="gpt-4",
        temperature=0.5,
        binary=binary
    )

    findings = await analyze_binary(binary)

    dn.log_metric("findings", len(findings))
```

## Installation

We publish every version to PyPi:
```bash
pip install -U dreadnode
```

If you want to build from source:

```bash

poetry install
# Install with multimodal extras
poetry install --extras multimodal

# Install with training extras
poetry install --extras training

# Install with all extras
poetry install --all-extras
```

## Installation from PyPI with Optional Features

For advanced media processing capabilities (audio, video, images), install the multimodal extras:

```bash
# Multimodal support (audio, video processing)
pip install -U "dreadnode[multimodal]"

# Training support (ML model integration)
pip install -U "dreadnode[training]"

# All optional features
pip install -U "dreadnode[all]"
```

See our **[installation guide](https://docs.dreadnode.io/strikes/install)** for more options.

## Getting Started

Read through our **[introduction guide](https://docs.dreadnode.io/strikes/intro)** in the docs.

## Examples

Check out **[dreadnode/example-agents](https://github.com/dreadnode/example-agents)** to find your favorite use case.
