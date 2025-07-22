# AI Red Teaming Module - Technical Design Document

## Overview

This document provides detailed technical specifications for implementing the AI Red Teaming module within the Dreadnode SDK. It follows the established architectural patterns, coding style, and type safety requirements of the existing codebase.

## Module Architecture

### Package Structure
```
dreadnode/
├── red_team/
│   ├── __init__.py              # Public API exports
│   ├── types.py                 # Type definitions and enums
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py              # Base classes and protocols
│   │   ├── metrics.py           # Scoring and evaluation metrics
│   │   ├── campaign.py          # Attack campaign orchestration
│   │   └── utils.py             # Shared utilities and helpers
│   ├── evasion/
│   │   ├── __init__.py
│   │   ├── attacks.py           # Adversarial attack implementations
│   │   ├── generators.py        # Example generation algorithms
│   │   ├── evaluators.py        # Success evaluation logic
│   │   └── data_types.py        # Evasion-specific data models
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractors.py        # Model extraction algorithms  
│   │   ├── surrogates.py        # Surrogate model implementations
│   │   ├── validators.py        # Extraction validation logic
│   │   └── data_types.py        # Extraction-specific data models
│   ├── inversion/
│   │   ├── __init__.py
│   │   ├── inverters.py         # Data reconstruction algorithms
│   │   ├── queries.py           # Query strategy implementations
│   │   ├── analyzers.py         # Privacy analysis tools
│   │   └── data_types.py        # Inversion-specific data models
│   └── integrations/
│       ├── __init__.py
│       ├── http_client.py       # HTTP API interaction utilities
│       └── burp.py              # Burp Suite integration helpers
```

## Type System Design

### Core Type Definitions

```python
# dreadnode/red_team/types.py
import typing as t
from enum import Enum
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from dreadnode.types import AnyDict, JsonValue

# Enumerations following Literal pattern from main codebase
DataType = t.Literal["text", "image", "audio", "video", "tabular"]
AttackType = t.Literal["targeted", "untargeted", "semi_targeted"] 
AttackMethod = t.Literal["gradient_based", "query_based", "transfer", "genetic"]
ExtractionMethod = t.Literal["knockoff", "functionally_equivalent", "model_stealing"]
InversionMethod = t.Literal["gradient_based", "membership_inference", "model_inversion"]

# Success metrics following Dreadnode metric patterns
AttackSuccessMetrics = t.TypedDict("AttackSuccessMetrics", {
    "success_rate": float,
    "avg_confidence": float,
    "query_count": int,
    "execution_time": float,
})

# Configuration types
class RedTeamConfig(BaseModel):
    """Base configuration for red team operations."""
    
    max_queries: int = Field(default=1000, ge=1, le=100000)
    timeout: float = Field(default=300.0, ge=1.0)
    black_box_only: bool = True
    retry_attempts: int = Field(default=3, ge=0, le=10)
    rate_limit_delay: float = Field(default=1.0, ge=0.0)
    
class EvasionConfig(RedTeamConfig):
    """Configuration specific to evasion attacks."""
    
    attack_method: AttackMethod = "query_based"
    perturbation_budget: float = Field(default=0.1, ge=0.0, le=1.0)
    num_examples: int = Field(default=10, ge=1, le=1000)
    target_labels: list[str] | None = None

class ExtractionConfig(RedTeamConfig):
    """Configuration specific to model extraction."""
    
    extraction_method: ExtractionMethod = "knockoff"
    query_budget: int = Field(default=10000, ge=100, le=1000000)
    fidelity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    surrogate_architecture: str | None = None

class InversionConfig(RedTeamConfig):
    """Configuration specific to data inversion."""
    
    inversion_method: InversionMethod = "membership_inference"
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    privacy_categories: list[str] = Field(default_factory=list)
    sample_count: int = Field(default=100, ge=1, le=10000)

# Generic type variables following Dreadnode patterns
P = t.ParamSpec("P")
R = t.TypeVar("R")
AttackResult = t.TypeVar("AttackResult")
```

### Data Models

```python
# Following Dreadnode's dataclass + pydantic hybrid approach
from dataclasses import dataclass, field
from datetime import datetime, timezone
from ulid import ULID

@dataclass
class AdversarialExample:
    """Represents a generated adversarial example."""
    
    id: str = field(default_factory=lambda: str(ULID()))
    original_input: t.Any
    adversarial_input: t.Any
    target_label: str | None = None
    predicted_label: str | None = None
    confidence: float = 0.0
    successful: bool = False
    perturbation_magnitude: float = 0.0
    generation_method: AttackMethod = "query_based"
    query_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: AnyDict = field(default_factory=dict)

@dataclass  
class ExtractedModel:
    """Represents an extracted model and its performance."""
    
    id: str = field(default_factory=lambda: str(ULID()))
    extraction_method: ExtractionMethod
    query_count: int
    fidelity_score: float
    accuracy_score: float | None = None
    model_artifact_path: Path | None = None
    function_signature: str | None = None
    extraction_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: AnyDict = field(default_factory=dict)

@dataclass
class ReconstructedSample:  
    """Represents reconstructed training data."""
    
    id: str = field(default_factory=lambda: str(ULID()))
    reconstructed_data: t.Any
    confidence: float
    privacy_category: str | None = None
    reconstruction_method: InversionMethod
    query_count: int
    similarity_score: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: AnyDict = field(default_factory=dict)
```

## Base Architecture

### Protocol Definitions

```python
# dreadnode/red_team/core/base.py
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

if t.TYPE_CHECKING:
    from opentelemetry.trace import Tracer
    from dreadnode.tracing.span import TaskSpan

# Protocol definitions following Dreadnode patterns
class RedTeamTarget(t.Protocol):
    """Protocol for red team attack targets."""
    
    endpoint: str
    api_key: str | None
    headers: AnyDict | None
    
    async def query(self, input_data: t.Any) -> t.Any: ...
    async def batch_query(self, inputs: list[t.Any]) -> list[t.Any]: ...

class AttackGenerator(t.Protocol[AttackResult]):
    """Protocol for attack generation algorithms."""
    
    async def generate(
        self, 
        target: RedTeamTarget, 
        config: RedTeamConfig,
    ) -> AttackResult: ...
    
    def estimate_cost(self, config: RedTeamConfig) -> int: ...

class AttackEvaluator(t.Protocol[AttackResult]):
    """Protocol for evaluating attack success."""
    
    async def evaluate(
        self, 
        attack_result: AttackResult,
        ground_truth: t.Any | None = None,
    ) -> AttackSuccessMetrics: ...

# Base classes following Dreadnode Task pattern
@dataclass
class RedTeamTask(t.Generic[P, R], ABC):
    """Base class for all red team tasks."""
    
    # Core attributes following Task pattern
    tracer: "Tracer"
    name: str
    attributes: AnyDict
    tags: list[str] = field(default_factory=list)
    
    # Red team specific attributes  
    target: RedTeamTarget | None = None
    config: RedTeamConfig = field(default_factory=RedTeamConfig)
    
    # Dreadnode integration
    log_inputs: bool = True
    log_outputs: bool = True
    log_artifacts: bool = True
    
    @abstractmethod
    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the red team attack."""
        ...
    
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute with full Dreadnode integration."""
        # Implementation follows Task.__call__ pattern
        with self._create_task_span() as span:
            if self.log_inputs:
                self._log_inputs(span, *args, **kwargs)
                
            try:
                result = await self.execute(*args, **kwargs)
                
                if self.log_outputs:
                    self._log_outputs(span, result)
                    
                if self.log_artifacts:
                    await self._log_artifacts(span, result)
                    
                return result
            except Exception as e:
                span.set_status("failed", str(e))
                raise
    
    def _create_task_span(self) -> "TaskSpan[R]":
        """Create task span following Dreadnode patterns."""
        from dreadnode.tracing.span import TaskSpan, current_run_span
        
        run = current_run_span.get()
        return TaskSpan(
            name=self.name,
            label=self.name.replace("_", " ").title(),
            attributes=self.attributes,
            tags=self.tags,
            run_id=run.run_id if run else "",
            tracer=self.tracer,
        )
```

### HTTP Target Implementation

```python
# dreadnode/red_team/integrations/http_client.py
import asyncio
import json
from typing import Any
from urllib.parse import urljoin

import httpx
from dreadnode.util import logger, handle_internal_errors

class HTTPTarget:
    """HTTP-based attack target implementation."""
    
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        headers: AnyDict | None = None,
        timeout: float = 30.0,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        
        # Build headers
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": f"dreadnode-red-team/{VERSION}",
        }
        if headers:
            self._headers.update(headers)
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
            
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers=self._headers,
        )
    
    @handle_internal_errors()
    async def query(self, input_data: Any) -> Any:
        """Send single query to target."""
        await asyncio.sleep(self.rate_limit_delay)
        
        try:
            response = await self._client.post(
                self.endpoint,
                json={"input": input_data}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    async def batch_query(self, inputs: list[Any]) -> list[Any]:
        """Send batch queries with rate limiting."""
        results = []
        for input_data in inputs:
            result = await self.query(input_data)
            results.append(result)
        return results
    
    async def __aenter__(self) -> "HTTPTarget":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._client.aclose()
```

## Attack Implementations

### Evasion Module

```python
# dreadnode/red_team/evasion/attacks.py
import random
import string
from typing import Any, Sequence

from dreadnode.util import handle_internal_errors
from ..core.base import RedTeamTask, AttackGenerator
from ..types import EvasionConfig, AdversarialExample, DataType

class TextEvasionGenerator:
    """Generates adversarial text examples."""
    
    def __init__(self, config: EvasionConfig) -> None:
        self.config = config
    
    async def generate_synonym_substitution(
        self,
        target: RedTeamTarget, 
        original_text: str,
        target_label: str | None = None,
    ) -> list[AdversarialExample]:
        """Generate adversarial examples using synonym substitution."""
        examples: list[AdversarialExample] = []
        
        # Simple synonym substitution strategy
        words = original_text.split()
        
        for _ in range(self.config.num_examples):
            # Randomly substitute words (simplified implementation)
            modified_words = words.copy()
            num_substitutions = max(1, len(words) // 10)
            
            for _ in range(num_substitutions):
                if words:
                    idx = random.randint(0, len(words) - 1)
                    # Simple character-level perturbation
                    word = modified_words[idx]
                    if len(word) > 1:
                        # Insert random character  
                        pos = random.randint(0, len(word))
                        char = random.choice(string.ascii_lowercase)
                        modified_words[idx] = word[:pos] + char + word[pos:]
            
            adversarial_text = " ".join(modified_words)
            
            # Query target with adversarial example
            try:
                response = await target.query(adversarial_text)
                predicted_label = self._extract_prediction(response)
                
                # Determine if attack was successful
                successful = (
                    target_label is None or  # Untargeted attack
                    predicted_label != target_label  # Targeted attack
                )
                
                example = AdversarialExample(
                    original_input=original_text,
                    adversarial_input=adversarial_text,
                    target_label=target_label,
                    predicted_label=predicted_label,
                    successful=successful,
                    generation_method="query_based",
                    query_count=1,
                )
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Failed to query target: {e}")
                continue
        
        return examples
    
    def _extract_prediction(self, response: Any) -> str | None:
        """Extract prediction from API response."""
        # Handle common response formats
        if isinstance(response, dict):
            return (
                response.get("prediction") or
                response.get("label") or  
                response.get("class") or
                response.get("output")
            )
        return str(response) if response else None

# Task implementation following Dreadnode patterns
class EvasionTask(RedTeamTask[..., list[AdversarialExample]]):
    """Evasion attack task implementation."""
    
    def __init__(
        self,
        tracer: "Tracer",
        *,
        name: str = "evasion_attack",
        target: RedTeamTarget,
        config: EvasionConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tracer=tracer,
            name=name,
            target=target,
            config=config,
            attributes={"attack_type": "evasion"},
            **kwargs,
        )
    
    @handle_internal_errors()
    async def execute(
        self,
        input_data: Any,
        data_type: DataType,
        target_labels: Sequence[str] | None = None,
    ) -> list[AdversarialExample]:
        """Execute evasion attack."""
        config = t.cast(EvasionConfig, self.config)
        
        if data_type == "text":
            generator = TextEvasionGenerator(config)
            return await generator.generate_synonym_substitution(
                self.target,
                input_data,
                target_labels[0] if target_labels else None,
            )
        else:
            raise NotImplementedError(f"Data type {data_type} not yet supported")

# High-level decorator following Dreadnode task decorator pattern  
def evasion_task(
    *,
    target: RedTeamTarget,
    config: EvasionConfig | None = None,
    scorers: Sequence[Scorer] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> t.Callable[[t.Callable], EvasionTask]:
    """Decorator for creating evasion tasks."""
    
    def decorator(func: t.Callable) -> EvasionTask:
        from dreadnode.main import DEFAULT_INSTANCE
        
        task_config = config or EvasionConfig()
        task_name = name or getattr(func, "__name__", "evasion_attack")
        
        return EvasionTask(
            tracer=DEFAULT_INSTANCE._get_tracer(),
            name=task_name,
            target=target,
            config=task_config,
            **kwargs,
        )
    
    return decorator
```

### Metrics and Scoring

```python
# dreadnode/red_team/core/metrics.py
from dreadnode.metric import Scorer, Metric
from ..types import AdversarialExample, ExtractedModel, ReconstructedSample

class EvasionSuccessScorer(Scorer[list[AdversarialExample]]):
    """Score evasion attack success rate."""
    
    def __init__(self, tracer: "Tracer", name: str = "evasion_success_rate") -> None:
        super().__init__(tracer, name)
    
    async def __call__(self, examples: list[AdversarialExample]) -> Metric:
        if not examples:
            return Metric(0.0)
        
        successful_count = sum(1 for ex in examples if ex.successful)
        success_rate = successful_count / len(examples)
        
        return Metric(
            value=success_rate,
            attributes={
                "total_examples": len(examples),
                "successful_examples": successful_count,
                "attack_type": "evasion",
            }
        )

class QueryEfficiencyScorer(Scorer[list[AdversarialExample]]):
    """Score query efficiency of attacks."""
    
    def __init__(self, tracer: "Tracer", name: str = "query_efficiency") -> None:
        super().__init__(tracer, name)
    
    async def __call__(self, examples: list[AdversarialExample]) -> Metric:
        if not examples:
            return Metric(0.0)
        
        successful_examples = [ex for ex in examples if ex.successful]
        if not successful_examples:
            return Metric(0.0)
        
        avg_queries = sum(ex.query_count for ex in successful_examples) / len(successful_examples)
        # Lower is better, so invert the score
        efficiency_score = 1.0 / max(1.0, avg_queries / 10.0)
        
        return Metric(
            value=efficiency_score,
            attributes={
                "avg_queries_per_success": avg_queries,
                "total_successful": len(successful_examples),
            }
        )

class ExtractionFidelityScorer(Scorer[ExtractedModel]):
    """Score model extraction fidelity."""
    
    def __init__(self, tracer: "Tracer", name: str = "extraction_fidelity") -> None:
        super().__init__(tracer, name)
    
    async def __call__(self, model: ExtractedModel) -> Metric:
        return Metric(
            value=model.fidelity_score,
            attributes={
                "query_count": model.query_count,
                "extraction_method": model.extraction_method,
                "accuracy_score": model.accuracy_score,
            }
        )
```

## Usage Examples Following Dreadnode Patterns

### Evasion Attack Example

```python
import dreadnode as dn
from dreadnode.red_team import (
    HTTPTarget, 
    EvasionConfig, 
    evasion_task,
    EvasionSuccessScorer,
    QueryEfficiencyScorer,
)

# Configure Dreadnode
dn.configure(
    server="https://api.dreadnode.io",
    token="your-api-token",
    project="ai-red-team-testing"
)

# Set up target and scorers
target = HTTPTarget(
    endpoint="https://api.example.com/classify",
    api_key="target-api-key",
    rate_limit_delay=0.5,
)

config = EvasionConfig(
    attack_method="query_based",
    num_examples=50,
    max_queries=1000,
)

success_scorer = EvasionSuccessScorer(dn.DEFAULT_INSTANCE._get_tracer())
efficiency_scorer = QueryEfficiencyScorer(dn.DEFAULT_INSTANCE._get_tracer())

# Create evasion task
@evasion_task(
    target=target,
    config=config,
    scorers=[success_scorer, efficiency_scorer],
    name="sentiment_evasion_test",
)
async def test_sentiment_classifier():
    """Test sentiment classifier robustness."""
    pass

# Execute within Dreadnode run
with dn.run("evasion-campaign-2024"):
    dn.log_params(
        target_model="sentiment-api-v2",
        attack_type="untargeted",
        data_type="text",
        num_examples=config.num_examples,
    )
    
    # Execute attack
    results = await test_sentiment_classifier(
        input_data="This movie was absolutely fantastic!",
        data_type="text",
        target_labels=["positive"],
    )
    
    # Additional metrics
    dn.log_metric("total_examples", len(results))
    dn.log_metric("avg_perturbation", 
                 sum(ex.perturbation_magnitude for ex in results) / len(results))
    
    # Log artifacts
    dn.log_artifact("adversarial_examples.json")
```

### Campaign Orchestration

```python
# dreadnode/red_team/core/campaign.py
@dataclass
class RedTeamCampaign:
    """Orchestrates multiple red team attacks."""
    
    name: str
    targets: list[RedTeamTarget]
    attacks: list[RedTeamTask] = field(default_factory=list)
    parallel_execution: bool = True
    
    async def execute(self) -> "CampaignResults":
        """Execute all attacks in the campaign."""
        results = []
        
        with dn.run(self.name):
            dn.log_params(
                campaign_name=self.name,
                num_targets=len(self.targets),
                num_attacks=len(self.attacks),
                parallel_execution=self.parallel_execution,
            )
            
            if self.parallel_execution:
                # Execute attacks concurrently
                tasks = []
                for attack in self.attacks:
                    for target in self.targets:
                        task = asyncio.create_task(attack(target))
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Execute attacks sequentially
                for attack in self.attacks:
                    for target in self.targets:
                        try:
                            result = await attack(target)
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Attack {attack.name} failed: {e}")
                            results.append(e)
            
            # Log campaign summary
            successful_attacks = sum(1 for r in results if not isinstance(r, Exception))
            dn.log_metric("campaign_success_rate", successful_attacks / len(results))
            
            return CampaignResults(
                campaign_name=self.name,
                results=results,
                success_rate=successful_attacks / len(results),
            )

# Usage
campaign = RedTeamCampaign(
    name="multi-model-evasion-test",
    targets=[
        HTTPTarget("https://api.model1.com/classify"),
        HTTPTarget("https://api.model2.com/classify"),  
        HTTPTarget("https://api.model3.com/classify"),
    ],
    attacks=[evasion_task, extraction_task, inversion_task],
)

results = await campaign.execute()
```

This technical design maintains consistency with the Dreadnode SDK's architectural patterns while providing comprehensive AI red teaming capabilities. The implementation follows the established type safety requirements, error handling patterns, and integration approaches found throughout the codebase.

## Next Steps

1. **Implementation Phase**: Begin with core infrastructure and evasion module
2. **Testing Framework**: Develop comprehensive test suites following existing patterns  
3. **Documentation**: Create usage examples and API documentation
4. **Integration Testing**: Validate compatibility with existing Dreadnode features
5. **Performance Optimization**: Ensure scalable execution for large campaigns