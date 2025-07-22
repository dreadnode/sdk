# AI Red Teaming API Reference

## Overview

The AI Red Teaming module provides a comprehensive API for conducting adversarial testing against AI/ML systems. It follows Dreadnode's established patterns for task management, experiment tracking, and result logging.

## Quick Start

```python
import dreadnode as dn
from dreadnode.red_team import HTTPTarget, EvasionConfig, evasion_attack

# Configure Dreadnode
dn.configure(
    server="https://api.dreadnode.io",
    token="your-api-token", 
    project="ai-red-team"
)

# Set up target
target = HTTPTarget("https://api.example.com/classify")

# Execute evasion attack
with dn.run("quick-evasion-test"):
    results = await evasion_attack(
        target=target,
        input_data="This is a test message",
        data_type="text",
        config=EvasionConfig(num_examples=10)
    )
    
    print(f"Generated {len(results)} adversarial examples")
    success_rate = sum(1 for r in results if r.successful) / len(results)
    print(f"Attack success rate: {success_rate:.2%}")
```

## Core Components

### Target Systems

#### HTTPTarget

HTTP-based target for API endpoints.

```python
class HTTPTarget:
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        rate_limit_delay: float = 1.0,
    ) -> None: ...
    
    async def query(self, input_data: Any) -> Any: ...
    async def batch_query(self, inputs: list[Any]) -> list[Any]: ...
```

**Parameters:**
- `endpoint`: Target API endpoint URL
- `api_key`: Optional API key for authentication
- `headers`: Additional HTTP headers
- `timeout`: Request timeout in seconds
- `rate_limit_delay`: Delay between requests in seconds

**Example:**
```python
target = HTTPTarget(
    endpoint="https://api.openai.com/v1/completions",
    api_key="sk-...",
    headers={"User-Agent": "Red-Team-Test/1.0"},
    rate_limit_delay=2.0,
)
```

### Configuration Classes

#### EvasionConfig

Configuration for evasion attacks.

```python
class EvasionConfig(BaseModel):
    attack_method: AttackMethod = "query_based"
    perturbation_budget: float = 0.1
    num_examples: int = 10
    target_labels: list[str] | None = None
    max_queries: int = 1000
    timeout: float = 300.0
    retry_attempts: int = 3
```

**Parameters:**
- `attack_method`: Method for generating adversarial examples
- `perturbation_budget`: Maximum allowed perturbation (0.0-1.0)
- `num_examples`: Number of adversarial examples to generate
- `target_labels`: Target labels for targeted attacks
- `max_queries`: Maximum queries to target system
- `timeout`: Overall attack timeout in seconds
- `retry_attempts`: Number of retry attempts for failed queries

#### ExtractionConfig

Configuration for model extraction attacks.

```python
class ExtractionConfig(BaseModel):
    extraction_method: ExtractionMethod = "knockoff"
    query_budget: int = 10000
    fidelity_threshold: float = 0.8
    surrogate_architecture: str | None = None
    max_queries: int = 100000
    timeout: float = 3600.0
```

**Parameters:**
- `extraction_method`: Model extraction algorithm
- `query_budget`: Total queries available for extraction
- `fidelity_threshold`: Minimum fidelity score for success
- `surrogate_architecture`: Architecture for surrogate model
- `max_queries`: Hard limit on total queries
- `timeout`: Maximum extraction time

#### InversionConfig

Configuration for data inversion attacks.

```python
class InversionConfig(BaseModel):
    inversion_method: InversionMethod = "membership_inference"
    confidence_threshold: float = 0.8
    privacy_categories: list[str] = []
    sample_count: int = 100
    max_queries: int = 10000
```

**Parameters:**
- `inversion_method`: Data inversion algorithm
- `confidence_threshold`: Minimum confidence for reconstructed data
- `privacy_categories`: Types of private information to target
- `sample_count`: Number of samples to attempt reconstruction
- `max_queries`: Maximum queries for inversion

## High-Level Attack Functions

### Evasion Attacks

#### evasion_attack

Generate adversarial examples against a target model.

```python
@dn.task(scorers=[EvasionSuccessScorer(), QueryEfficiencyScorer()])
async def evasion_attack(
    target: HTTPTarget,
    input_data: Any,
    data_type: DataType,
    *,
    config: EvasionConfig | None = None,
    attack_type: AttackType = "untargeted",
    target_labels: list[str] | None = None,
) -> list[AdversarialExample]: ...
```

**Parameters:**
- `target`: Target system to attack
- `input_data`: Original input to create adversarial examples from
- `data_type`: Type of input data ("text", "image", "audio", "video")
- `config`: Attack configuration (uses defaults if None)
- `attack_type`: "targeted", "untargeted", or "semi_targeted"
- `target_labels`: Labels to target (required for targeted attacks)

**Returns:**
- `list[AdversarialExample]`: Generated adversarial examples with success metrics

**Example:**
```python
with dn.run("text-evasion-test"):
    results = await evasion_attack(
        target=HTTPTarget("https://api.example.com/sentiment"),
        input_data="I love this product!",
        data_type="text",
        attack_type="targeted", 
        target_labels=["negative"],
        config=EvasionConfig(
            num_examples=25,
            attack_method="query_based",
            perturbation_budget=0.2,
        )
    )
    
    # Results are automatically logged with metrics
    successful = [ex for ex in results if ex.successful]
    print(f"Successfully generated {len(successful)} adversarial examples")
```

#### batch_evasion_attack

Generate adversarial examples for multiple inputs.

```python
@dn.task(scorers=[BatchEvasionScorer()])
async def batch_evasion_attack(
    target: HTTPTarget,
    input_batch: list[Any],
    data_type: DataType,
    *,
    config: EvasionConfig | None = None,
    attack_type: AttackType = "untargeted",
) -> list[list[AdversarialExample]]: ...
```

### Model Extraction Attacks

#### extract_model

Extract model functionality through black-box querying.

```python
@dn.task(scorers=[ExtractionFidelityScorer(), ExtractionEfficiencyScorer()])
async def extract_model(
    target: HTTPTarget,
    data_type: DataType,
    function_spec: str,
    *,
    config: ExtractionConfig | None = None,
    validation_data: list[tuple[Any, Any]] | None = None,
) -> ExtractedModel: ...
```

**Parameters:**
- `target`: Target model to extract
- `data_type`: Type of input data the model handles
- `function_spec`: Description of target model's function
- `config`: Extraction configuration
- `validation_data`: Optional validation data for fidelity testing

**Returns:**
- `ExtractedModel`: Extracted model with performance metrics

**Example:**
```python
with dn.run("sentiment-extraction"):
    extracted = await extract_model(
        target=HTTPTarget("https://api.example.com/sentiment"),
        data_type="text",
        function_spec="binary sentiment classification (positive/negative)",
        config=ExtractionConfig(
            query_budget=5000,
            fidelity_threshold=0.85,
            extraction_method="knockoff",
        ),
        validation_data=[(text, label) for text, label in test_set],
    )
    
    print(f"Extraction fidelity: {extracted.fidelity_score:.3f}")
    print(f"Queries used: {extracted.query_count}")
    
    # Extracted model is saved as artifact
    dn.log_artifact("extracted_model.pkl")
```

### Data Inversion Attacks

#### invert_training_data

Attempt to reconstruct training data from model responses.

```python
@dn.task(scorers=[ReconstructionQualityScorer(), PrivacyLeakageScorer()])
async def invert_training_data(
    target: HTTPTarget,
    data_type: DataType,
    *,
    config: InversionConfig | None = None,
    seed_data: list[Any] | None = None,
    privacy_categories: list[str] | None = None,
) -> list[ReconstructedSample]: ...
```

**Parameters:**
- `target`: Target model to attack
- `data_type`: Type of training data to reconstruct
- `config`: Inversion attack configuration
- `seed_data`: Optional seed data to guide reconstruction
- `privacy_categories`: Specific types of private info to target

**Returns:**
- `list[ReconstructedSample]`: Reconstructed training samples with confidence

**Example:**
```python
with dn.run("privacy-inversion-test"):
    reconstructed = await invert_training_data(
        target=HTTPTarget("https://api.example.com/embeddings"),
        data_type="text",
        config=InversionConfig(
            inversion_method="membership_inference",
            sample_count=200,
            confidence_threshold=0.75,
        ),
        privacy_categories=["email", "phone", "ssn"],
    )
    
    high_confidence = [s for s in reconstructed if s.confidence > 0.8]
    print(f"High-confidence reconstructions: {len(high_confidence)}")
    
    # Privacy analysis automatically logged
    for category in ["email", "phone", "ssn"]:
        category_samples = [s for s in reconstructed if s.privacy_category == category]
        dn.log_metric(f"privacy_leakage_{category}", len(category_samples))
```

## Data Models

### AdversarialExample

Represents a generated adversarial example.

```python
@dataclass
class AdversarialExample:
    id: str                                    # Unique identifier
    original_input: Any                        # Original input data
    adversarial_input: Any                     # Perturbed input data
    target_label: str | None                   # Target label (if targeted attack)
    predicted_label: str | None                # Model's prediction
    confidence: float                          # Attack confidence score
    successful: bool                           # Whether attack succeeded
    perturbation_magnitude: float              # Size of perturbation
    generation_method: AttackMethod            # Method used to generate
    query_count: int                          # Queries used for generation
    timestamp: datetime                       # Creation timestamp
    metadata: dict[str, Any]                  # Additional metadata
```

### ExtractedModel

Represents an extracted model and performance metrics.

```python
@dataclass
class ExtractedModel:
    id: str                                    # Unique identifier
    extraction_method: ExtractionMethod        # Extraction algorithm used
    query_count: int                          # Total queries used
    fidelity_score: float                     # Similarity to original model
    accuracy_score: float | None              # Accuracy on test data
    model_artifact_path: Path | None          # Path to saved model
    function_signature: str | None            # Inferred function signature
    extraction_time: float                    # Time taken for extraction
    timestamp: datetime                       # Extraction timestamp
    metadata: dict[str, Any]                  # Additional metadata
```

### ReconstructedSample

Represents reconstructed training data.

```python
@dataclass
class ReconstructedSample:
    id: str                                    # Unique identifier
    reconstructed_data: Any                    # Reconstructed data sample
    confidence: float                          # Reconstruction confidence
    privacy_category: str | None               # Type of private information
    reconstruction_method: InversionMethod     # Method used
    query_count: int                          # Queries used for reconstruction
    similarity_score: float | None            # Similarity to real data
    timestamp: datetime                       # Reconstruction timestamp
    metadata: dict[str, Any]                  # Additional metadata
```

## Campaign Management

### RedTeamCampaign

Orchestrate multiple red team attacks.

```python
@dataclass
class RedTeamCampaign:
    name: str                                  # Campaign name
    targets: list[HTTPTarget]                  # Target systems
    attacks: list[Callable]                   # Attack functions to execute
    parallel_execution: bool = True            # Execute attacks in parallel
    
    async def execute(self) -> CampaignResults: ...
```

**Example:**
```python
# Define targets
targets = [
    HTTPTarget("https://api.model1.com/classify"),
    HTTPTarget("https://api.model2.com/classify"), 
    HTTPTarget("https://api.model3.com/classify"),
]

# Create campaign
campaign = RedTeamCampaign(
    name="multi-model-robustness-test",
    targets=targets,
    attacks=[
        partial(evasion_attack, 
                input_data="Test message",
                data_type="text"),
        partial(extract_model,
                data_type="text", 
                function_spec="sentiment classification"),
    ],
)

# Execute campaign
with dn.run("robustness-campaign-2024"):
    results = await campaign.execute()
    
    print(f"Campaign success rate: {results.success_rate:.2%}")
    print(f"Total attacks executed: {len(results.results)}")
```

## Scoring and Metrics

### Built-in Scorers

#### EvasionSuccessScorer

Scores evasion attack success rate.

```python
scorer = EvasionSuccessScorer()
score = await scorer(adversarial_examples)
print(f"Success rate: {score.value:.2%}")
```

#### QueryEfficiencyScorer

Scores query efficiency (lower queries = higher score).

```python
scorer = QueryEfficiencyScorer()
score = await scorer(adversarial_examples)
print(f"Query efficiency: {score.value:.3f}")
```

#### ExtractionFidelityScorer

Scores model extraction fidelity.

```python
scorer = ExtractionFidelityScorer()  
score = await scorer(extracted_model)
print(f"Extraction fidelity: {score.value:.3f}")
```

#### ReconstructionQualityScorer

Scores data reconstruction quality.

```python
scorer = ReconstructionQualityScorer()
score = await scorer(reconstructed_samples)
print(f"Reconstruction quality: {score.value:.3f}")
```

### Custom Scorers

Create custom scorers following Dreadnode patterns:

```python
class CustomAttackScorer(dn.Scorer[list[AdversarialExample]]):
    def __init__(self, tracer, name: str = "custom_attack_score"):
        super().__init__(tracer, name)
    
    async def __call__(self, examples: list[AdversarialExample]) -> dn.Metric:
        # Custom scoring logic
        custom_score = self.compute_custom_score(examples)
        
        return dn.Metric(
            value=custom_score,
            attributes={
                "total_examples": len(examples),
                "scoring_method": "custom",
            }
        )
    
    def compute_custom_score(self, examples: list[AdversarialExample]) -> float:
        # Your custom scoring implementation
        return 0.85

# Use custom scorer
@dn.task(scorers=[CustomAttackScorer(dn.DEFAULT_INSTANCE._get_tracer())])
async def custom_evasion_attack(...) -> list[AdversarialExample]:
    # Attack implementation
    pass
```

## Integration Patterns

### With Rigging (LLM Testing)

```python
import rigging as rg
import dreadnode as dn
from dreadnode.red_team import evasion_attack, HTTPTarget

# Combined LLM + Red Team evaluation
@dn.task(scorers=[dn.red_team.EvasionSuccessScorer(), SafetyScorer()])
@rg.prompt(generator_id="gpt-4o")
async def test_llm_safety_with_adversarial(
    base_prompt: str,
    safety_categories: list[str],
) -> SafetyTestResults:
    """Test LLM safety using both prompt engineering and adversarial examples."""
    
    # First test with rigging
    safety_result = await test_safety_with_rigging(base_prompt)
    
    # Then test with adversarial examples
    target = HTTPTarget("https://api.openai.com/v1/completions")
    adversarial_results = await evasion_attack(
        target=target,
        input_data=base_prompt,
        data_type="text",
        attack_type="targeted",
        target_labels=safety_categories,
    )
    
    return SafetyTestResults(
        rigging_results=safety_result,
        adversarial_results=adversarial_results,
    )
```

### Burp Suite Integration

```python
from dreadnode.red_team.integrations.burp import BurpTarget

# Use Burp Suite as attack proxy
burp_target = BurpTarget(
    burp_proxy="http://127.0.0.1:8080",
    target_url="https://api.target.com/ml",
)

# Attacks will go through Burp for inspection
with dn.run("burp-integrated-test"):
    results = await evasion_attack(
        target=burp_target,
        input_data="Test input",
        data_type="text",
    )
```

## Error Handling

All red team functions follow Dreadnode's error handling patterns:

```python
try:
    with dn.run("error-handling-example"):
        results = await evasion_attack(
            target=HTTPTarget("https://invalid-endpoint.com"),
            input_data="test",
            data_type="text",
        )
except RedTeamError as e:
    # Red team specific errors
    print(f"Red team error: {e}")
except dn.DreadnodeUsageWarning as e:
    # Dreadnode usage warnings
    print(f"Usage warning: {e}")
except Exception as e:
    # General errors are automatically handled and logged
    print(f"Unexpected error: {e}")
```

## Rate Limiting and Ethics

### Rate Limiting

```python
# Built-in rate limiting
target = HTTPTarget(
    endpoint="https://api.example.com/classify",
    rate_limit_delay=2.0,  # 2 second delay between requests
)

# Advanced rate limiting
config = EvasionConfig(
    max_queries=100,        # Hard limit on queries
    timeout=300.0,         # Overall timeout
    retry_attempts=3,      # Retry failed queries
)
```

### Responsible Testing

```python
# Authorization checking (recommended pattern)
class AuthorizedTarget(HTTPTarget):
    def __init__(self, endpoint: str, authorization_token: str, **kwargs):
        super().__init__(endpoint, **kwargs)
        self.authorization_token = authorization_token
    
    async def query(self, input_data: Any) -> Any:
        # Verify authorization before each query
        if not self._verify_authorization():
            raise ValueError("Unauthorized target access")
        return await super().query(input_data)
    
    def _verify_authorization(self) -> bool:
        # Implement authorization verification
        return self.authorization_token == "authorized-testing-token"

# Use with explicit authorization
authorized_target = AuthorizedTarget(
    endpoint="https://internal-api.company.com/ml",
    authorization_token="authorized-testing-token",
)
```

This API reference provides comprehensive guidance for using the AI Red Teaming module while maintaining consistency with Dreadnode's established patterns and ensuring responsible usage.