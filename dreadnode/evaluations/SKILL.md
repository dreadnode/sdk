SKILL.md

# Evaluation Designer Skill

A skill for designing comprehensive evaluation pipelines in Dreadnode.

## Overview

The Evaluation Designer helps you:

- Design evaluation datasets
- Select appropriate scorers
- Configure evaluation parameters
- Build evaluation pipelines

## Usage

```python
import typing as t
from dataclasses import dataclass, field
from enum import Enum


class ScoreType(Enum):
    """Types of scoring approaches."""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    SIMILARITY = "similarity"
    LLM_JUDGE = "llm_judge"
    CUSTOM = "custom"


@dataclass
class DatasetConfig:
    """Configuration for evaluation dataset."""
    name: str
    input_fields: list[str]
    output_field: str = "output"
    reference_field: str = None
    size: int = 0
    source: str = None


@dataclass
class ScorerConfig:
    """Configuration for a scorer."""
    name: str
    score_type: ScoreType
    config: dict = field(default_factory=dict)
    weight: float = 1.0
    is_assertion: bool = False


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration."""
    name: str
    task_name: str
    dataset: DatasetConfig
    scorers: list[ScorerConfig]
    params: dict = field(default_factory=dict)
    iterations: int = 1
    concurrency: int = 5
    max_errors: int = None


class EvaluationDesigner:
    """
    Design evaluation pipelines with recommended configurations.
    """

    # Scorer templates
    SCORER_TEMPLATES = {
        "exact_match": {
            "type": ScoreType.EXACT_MATCH,
            "description": "Exact string match between output and reference",
            "requires_reference": True,
            "config": {}
        },
        "contains_keyword": {
            "type": ScoreType.CONTAINS,
            "description": "Check if output contains specific keywords",
            "requires_reference": False,
            "config": {"keywords": [], "case_sensitive": False}
        },
        "contains_reference": {
            "type": ScoreType.CONTAINS,
            "description": "Check if output contains the reference",
            "requires_reference": True,
            "config": {}
        },
        "semantic_similarity": {
            "type": ScoreType.SIMILARITY,
            "description": "Semantic similarity via embeddings",
            "requires_reference": True,
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "threshold": 0.7}
        },
        "llm_quality": {
            "type": ScoreType.LLM_JUDGE,
            "description": "LLM-based quality assessment",
            "requires_reference": False,
            "config": {"model": "gpt-4", "criteria": ["relevance", "accuracy", "clarity"]}
        },
        "llm_comparison": {
            "type": ScoreType.LLM_JUDGE,
            "description": "LLM comparison against reference",
            "requires_reference": True,
            "config": {"model": "gpt-4", "rubric": "Compare output to reference for accuracy"}
        },
        "length_check": {
            "type": ScoreType.CUSTOM,
            "description": "Check output length is within range",
            "requires_reference": False,
            "config": {"min_length": 10, "max_length": 1000}
        },
        "format_check": {
            "type": ScoreType.CUSTOM,
            "description": "Verify output format (JSON, XML, etc.)",
            "requires_reference": False,
            "config": {"format": "json"}
        }
    }

    # Evaluation templates by use case
    EVALUATION_TEMPLATES = {
        "qa_evaluation": {
            "description": "Evaluate question answering tasks",
            "scorers": ["exact_match", "contains_reference", "semantic_similarity"],
            "iterations": 1,
            "recommended_dataset_fields": ["question", "context", "answer"]
        },
        "summarization": {
            "description": "Evaluate text summarization",
            "scorers": ["semantic_similarity", "length_check", "llm_quality"],
            "iterations": 1,
            "recommended_dataset_fields": ["text", "summary"]
        },
        "classification": {
            "description": "Evaluate classification tasks",
            "scorers": ["exact_match"],
            "iterations": 3,
            "recommended_dataset_fields": ["text", "label"]
        },
        "generation": {
            "description": "Evaluate open-ended generation",
            "scorers": ["llm_quality", "length_check", "format_check"],
            "iterations": 3,
            "recommended_dataset_fields": ["prompt"]
        },
        "code_generation": {
            "description": "Evaluate code generation tasks",
            "scorers": ["exact_match", "contains_keyword"],
            "iterations": 1,
            "recommended_dataset_fields": ["prompt", "expected_output", "test_cases"]
        },
        "safety_evaluation": {
            "description": "Evaluate model safety and refusals",
            "scorers": ["contains_keyword", "llm_quality"],
            "iterations": 5,
            "recommended_dataset_fields": ["prompt", "expected_behavior"]
        }
    }

    def recommend_template(self, task_description: str) -> dict:
        """
        Recommend an evaluation template based on task description.

        Args:
            task_description: Description of what to evaluate

        Returns:
            Recommended template with configuration
        """
        task_lower = task_description.lower()

        # Keywords for each template
        template_keywords = {
            "qa_evaluation": ["question", "answer", "qa", "factual", "knowledge"],
            "summarization": ["summary", "summarize", "condense", "brief"],
            "classification": ["classify", "categorize", "label", "sentiment"],
            "generation": ["generate", "create", "write", "creative"],
            "code_generation": ["code", "program", "function", "script"],
            "safety_evaluation": ["safety", "harmful", "refusal", "jailbreak", "security"]
        }

        # Score each template
        scores = {}
        for template, keywords in template_keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                scores[template] = score

        if not scores:
            return {
                "recommendation": "generation",
                "confidence": 0.5,
                "reason": "Default template for general evaluation"
            }

        best = max(scores.items(), key=lambda x: x[1])

        return {
            "recommendation": best[0],
            "confidence": min(best[1] / 3.0, 1.0),
            "template": self.EVALUATION_TEMPLATES[best[0]],
            "alternatives": [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:3]]
        }

    def design_evaluation(
        self,
        name: str,
        task_name: str,
        template: str = None,
        custom_scorers: list[str] = None,
        dataset_fields: list[str] = None,
        iterations: int = None,
        params: dict = None
    ) -> EvaluationConfig:
        """
        Design a complete evaluation configuration.

        Args:
            name: Name for the evaluation
            task_name: Name of the task to evaluate
            template: Template name (or None for custom)
            custom_scorers: Additional scorer names
            dataset_fields: Input field names for dataset
            iterations: Number of iterations
            params: Grid search parameters

        Returns:
            Complete EvaluationConfig
        """
        # Get template
        if template and template in self.EVALUATION_TEMPLATES:
            tmpl = self.EVALUATION_TEMPLATES[template]
            scorer_names = tmpl["scorers"]
            default_iterations = tmpl["iterations"]
            default_fields = tmpl["recommended_dataset_fields"]
        else:
            scorer_names = []
            default_iterations = 1
            default_fields = ["input"]

        # Add custom scorers
        if custom_scorers:
            scorer_names = list(set(scorer_names + custom_scorers))

        # Build scorer configs
        scorers = []
        for name_ in scorer_names:
            if name_ in self.SCORER_TEMPLATES:
                tmpl = self.SCORER_TEMPLATES[name_]
                scorers.append(ScorerConfig(
                    name=name_,
                    score_type=tmpl["type"],
                    config=tmpl["config"].copy(),
                    weight=1.0,
                    is_assertion=(tmpl["type"] == ScoreType.EXACT_MATCH)
                ))

        # Build dataset config
        fields = dataset_fields or default_fields
        dataset = DatasetConfig(
            name=f"{name}_dataset",
            input_fields=fields[:-1] if len(fields) > 1 else fields,
            output_field="output",
            reference_field=fields[-1] if len(fields) > 1 else None
        )

        return EvaluationConfig(
            name=name,
            task_name=task_name,
            dataset=dataset,
            scorers=scorers,
            params=params or {},
            iterations=iterations or default_iterations,
            concurrency=5
        )

    def generate_code(self, config: EvaluationConfig) -> str:
        """
        Generate Python code for an evaluation configuration.

        Args:
            config: EvaluationConfig to generate code for

        Returns:
            Python code string
        """
        scorer_imports = []
        scorer_definitions = []

        for scorer in config.scorers:
            if scorer.score_type == ScoreType.EXACT_MATCH:
                scorer_imports.append("from dreadnode.scorers import equals")
                scorer_definitions.append(f"equals(TaskInput('{config.dataset.reference_field}')).with_(name='{scorer.name}')")
            elif scorer.score_type == ScoreType.CONTAINS:
                scorer_imports.append("from dreadnode.scorers import contains")
                if scorer.config.get("keywords"):
                    scorer_definitions.append(f"contains({scorer.config['keywords']!r})")
                else:
                    scorer_definitions.append(f"contains(TaskInput('{config.dataset.reference_field}'))")
            elif scorer.score_type == ScoreType.SIMILARITY:
                scorer_imports.append("from dreadnode.scorers import similarity")
                scorer_definitions.append(f"similarity(reference=TaskInput('{config.dataset.reference_field}'))")
            elif scorer.score_type == ScoreType.LLM_JUDGE:
                scorer_imports.append("from dreadnode.scorers import llm_judge")
                scorer_definitions.append(f"llm_judge(model='{scorer.config.get('model', 'gpt-4')}')")

        imports = list(set(scorer_imports))

        code = f'''"""
{config.name} Evaluation

Auto-generated evaluation configuration.
"""

from dreadnode import Evaluation, task
from dreadnode.core.meta import TaskInput
{chr(10).join(imports)}


@task(name="{config.task_name}")
async def {config.task_name}({', '.join(f'{f}: str' for f in config.dataset.input_fields)}) -> str:
    # TODO: Implement your task logic
    pass


# Define scorers
scorers = [
    {(',' + chr(10) + '    ').join(scorer_definitions)}
]

# Create evaluation
evaluation = Evaluation(
    name="{config.name}",
    task={config.task_name},
    dataset=None,  # Load your dataset here
    scorers=scorers,
    assert_scores={[s.name for s in config.scorers if s.is_assertion]!r},
    iterations={config.iterations},
    concurrency={config.concurrency},
)

# Run evaluation
if __name__ == "__main__":
    import asyncio
    result = asyncio.run(evaluation.run())
    print(f"Pass rate: {{result.pass_rate:.1%}}")
'''
        return code

    def list_templates(self) -> dict:
        """List all available templates."""
        return {
            "evaluation_templates": {
                name: info["description"]
                for name, info in self.EVALUATION_TEMPLATES.items()
            },
            "scorer_templates": {
                name: info["description"]
                for name, info in self.SCORER_TEMPLATES.items()
            }
        }


# Convenience functions
def design_evaluation(task: str, **kwargs) -> EvaluationConfig:
    """Quick evaluation design."""
    designer = EvaluationDesigner()
    rec = designer.recommend_template(task)
    return designer.design_evaluation(
        name=f"eval_{task.replace(' ', '_')[:20]}",
        task_name="my_task",
        template=rec["recommendation"],
        **kwargs
    )


# Example
if __name__ == "__main__":
    designer = EvaluationDesigner()

    # Recommend template
    rec = designer.recommend_template("Evaluate a QA system for factual accuracy")
    print(f"Recommended: {rec['recommendation']} (confidence: {rec['confidence']:.0%})")

    # Design evaluation
    config = designer.design_evaluation(
        name="qa_eval",
        task_name="answer_question",
        template="qa_evaluation",
        dataset_fields=["question", "context", "answer"]
    )

    # Generate code
    code = designer.generate_code(config)
    print("\nGenerated Code:")
    print(code)
```

## Templates

### Evaluation Templates

| Template          | Description           | Scorers                           |
| ----------------- | --------------------- | --------------------------------- |
| qa_evaluation     | Question answering    | exact_match, contains, similarity |
| summarization     | Text summarization    | similarity, length, llm_quality   |
| classification    | Classification tasks  | exact_match                       |
| generation        | Open-ended generation | llm_quality, length, format       |
| code_generation   | Code generation       | exact_match, contains             |
| safety_evaluation | Safety testing        | contains, llm_quality             |

### Scorer Templates

| Scorer              | Type        | Description                |
| ------------------- | ----------- | -------------------------- |
| exact_match         | EXACT_MATCH | Exact string match         |
| contains_keyword    | CONTAINS    | Keyword containment        |
| semantic_similarity | SIMILARITY  | Embedding-based similarity |
| llm_quality         | LLM_JUDGE   | LLM quality assessment     |
| length_check        | CUSTOM      | Output length validation   |
| format_check        | CUSTOM      | Format validation          |

## Integration

```python
from evaluation_designer import EvaluationDesigner, design_evaluation

# Quick design
config = design_evaluation("Evaluate summarization quality")

# Full workflow
designer = EvaluationDesigner()

# Get recommendation
rec = designer.recommend_template("Test QA accuracy")

# Design with customization
config = designer.design_evaluation(
    name="custom_eval",
    task_name="my_qa_task",
    template=rec["recommendation"],
    custom_scorers=["llm_comparison"],
    iterations=5
)

# Generate runnable code
code = designer.generate_code(config)
```
