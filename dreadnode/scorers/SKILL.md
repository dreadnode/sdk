# Scorer Composer Skill

A skill for composing complex scoring functions from simple building blocks.

## Overview

The Scorer Composer enables:

- Building composite scorers from primitives
- Visual scorer pipeline design
- Automatic weight optimization
- Scorer validation and testing

## Usage

```python
import typing as t
from dataclasses import dataclass
from enum import Enum


class CompositionOp(Enum):
    """Composition operations for scorers."""
    AVG = "avg"
    WEIGHTED_AVG = "weighted_avg"
    AND = "and"
    OR = "or"
    MIN = "min"
    MAX = "max"
    SCALE = "scale"
    CLIP = "clip"
    THRESHOLD = "threshold"
    INVERT = "invert"


@dataclass
class ScorerNode:
    """A node in the scorer composition tree."""
    name: str
    type: str  # "primitive" or "composite"
    operation: CompositionOp = None
    children: list["ScorerNode"] = None
    config: dict = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.config is None:
            self.config = {}


class ScorerComposer:
    """
    Compose complex scorers from building blocks.
    """

    # Primitive scorers
    PRIMITIVES = {
        "length": {
            "description": "Score based on text length",
            "config": {"normalize_to": 100}
        },
        "word_count": {
            "description": "Score based on word count",
            "config": {"normalize_to": 50}
        },
        "contains": {
            "description": "Check text containment",
            "config": {"target": "", "case_sensitive": False}
        },
        "regex_match": {
            "description": "Match regex pattern",
            "config": {"pattern": ""}
        },
        "sentiment": {
            "description": "Sentiment analysis score",
            "config": {}
        },
        "readability": {
            "description": "Text readability score",
            "config": {"metric": "flesch_kincaid"}
        },
        "similarity": {
            "description": "Semantic similarity",
            "config": {"model": "all-MiniLM-L6-v2"}
        },
        "llm_judge": {
            "description": "LLM-based scoring",
            "config": {"model": "gpt-4", "rubric": ""}
        },
        "custom": {
            "description": "Custom scoring function",
            "config": {"function": None}
        }
    }

    def __init__(self):
        self.scorers: dict[str, ScorerNode] = {}

    def create_primitive(
        self,
        name: str,
        primitive_type: str,
        **config
    ) -> ScorerNode:
        """
        Create a primitive scorer.

        Args:
            name: Name for the scorer
            primitive_type: Type from PRIMITIVES
            **config: Configuration overrides

        Returns:
            ScorerNode for the primitive
        """
        if primitive_type not in self.PRIMITIVES:
            raise ValueError(f"Unknown primitive: {primitive_type}")

        default_config = self.PRIMITIVES[primitive_type]["config"].copy()
        default_config.update(config)

        node = ScorerNode(
            name=name,
            type="primitive",
            config={"primitive": primitive_type, **default_config}
        )
        self.scorers[name] = node
        return node

    def compose(
        self,
        name: str,
        operation: CompositionOp,
        children: list[str | ScorerNode],
        **config
    ) -> ScorerNode:
        """
        Compose multiple scorers with an operation.

        Args:
            name: Name for the composite scorer
            operation: Composition operation
            children: Child scorers (names or nodes)
            **config: Operation-specific config

        Returns:
            ScorerNode for the composite
        """
        child_nodes = []
        for child in children:
            if isinstance(child, str):
                if child not in self.scorers:
                    raise ValueError(f"Unknown scorer: {child}")
                child_nodes.append(self.scorers[child])
            else:
                child_nodes.append(child)

        node = ScorerNode(
            name=name,
            type="composite",
            operation=operation,
            children=child_nodes,
            config=config
        )
        self.scorers[name] = node
        return node

    # Convenience methods for common operations
    def avg(self, name: str, children: list[str]) -> ScorerNode:
        """Average multiple scorers."""
        return self.compose(name, CompositionOp.AVG, children)

    def weighted_avg(
        self,
        name: str,
        children: list[str],
        weights: list[float]
    ) -> ScorerNode:
        """Weighted average of scorers."""
        return self.compose(
            name, CompositionOp.WEIGHTED_AVG, children, weights=weights
        )

    def all_pass(self, name: str, children: list[str]) -> ScorerNode:
        """All scorers must be truthy (AND)."""
        return self.compose(name, CompositionOp.AND, children)

    def any_pass(self, name: str, children: list[str]) -> ScorerNode:
        """Any scorer must be truthy (OR)."""
        return self.compose(name, CompositionOp.OR, children)

    def scale(
        self,
        name: str,
        child: str,
        factor: float
    ) -> ScorerNode:
        """Scale a scorer's output."""
        return self.compose(name, CompositionOp.SCALE, [child], factor=factor)

    def clip(
        self,
        name: str,
        child: str,
        min_val: float,
        max_val: float
    ) -> ScorerNode:
        """Clip scorer output to range."""
        return self.compose(
            name, CompositionOp.CLIP, [child],
            min_val=min_val, max_val=max_val
        )

    def threshold(
        self,
        name: str,
        child: str,
        threshold: float
    ) -> ScorerNode:
        """Apply binary threshold."""
        return self.compose(
            name, CompositionOp.THRESHOLD, [child], threshold=threshold
        )

    def invert(self, name: str, child: str) -> ScorerNode:
        """Invert scorer (1 - value)."""
        return self.compose(name, CompositionOp.INVERT, [child])

    def visualize(self, name: str) -> str:
        """
        Create ASCII visualization of scorer tree.

        Args:
            name: Name of scorer to visualize

        Returns:
            ASCII tree representation
        """
        if name not in self.scorers:
            return f"Unknown scorer: {name}"

        def draw_node(node: ScorerNode, prefix: str = "", is_last: bool = True) -> str:
            connector = "└── " if is_last else "├── "
            result = prefix + connector + f"{node.name}"

            if node.type == "primitive":
                result += f" [{node.config.get('primitive', 'custom')}]"
            else:
                result += f" ({node.operation.value})"

            result += "\n"

            if node.children:
                child_prefix = prefix + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    is_last_child = i == len(node.children) - 1
                    result += draw_node(child, child_prefix, is_last_child)

            return result

        return f"Scorer Tree: {name}\n" + draw_node(self.scorers[name], "", True)

    def generate_code(self, name: str) -> str:
        """
        Generate Python code for the scorer.

        Args:
            name: Name of scorer to generate code for

        Returns:
            Python code string
        """
        if name not in self.scorers:
            return f"# Unknown scorer: {name}"

        imports = set()
        code_lines = []

        def generate_node(node: ScorerNode, var_name: str) -> str:
            if node.type == "primitive":
                prim = node.config.get("primitive", "custom")

                if prim == "length":
                    imports.add("from dreadnode.scorers import length_in_range")
                    norm = node.config.get("normalize_to", 100)
                    return f"{var_name} = length_in_range(0, {norm})"

                elif prim == "contains":
                    imports.add("from dreadnode.scorers import contains")
                    target = node.config.get("target", "")
                    case = node.config.get("case_sensitive", False)
                    return f'{var_name} = contains("{target}", case_sensitive={case})'

                elif prim == "similarity":
                    imports.add("from dreadnode.scorers import similarity")
                    return f"{var_name} = similarity()"

                elif prim == "llm_judge":
                    imports.add("from dreadnode.scorers import llm_judge")
                    model = node.config.get("model", "gpt-4")
                    return f'{var_name} = llm_judge(model="{model}")'

                else:
                    return f"# {var_name} = custom_scorer  # TODO: implement"

            else:
                # Composite
                child_vars = []
                for i, child in enumerate(node.children):
                    child_var = f"{var_name}_child{i}"
                    code_lines.append(generate_node(child, child_var))
                    child_vars.append(child_var)

                op = node.operation
                if op == CompositionOp.AVG:
                    imports.add("from dreadnode.core.scorer import avg")
                    return f"{var_name} = avg([{', '.join(child_vars)}])"

                elif op == CompositionOp.WEIGHTED_AVG:
                    imports.add("from dreadnode.core.scorer import weighted_avg")
                    weights = node.config.get("weights", [1.0] * len(child_vars))
                    return f"{var_name} = weighted_avg([{', '.join(child_vars)}], {weights})"

                elif op == CompositionOp.AND:
                    imports.add("from dreadnode.core.scorer import and_")
                    return f"{var_name} = and_([{', '.join(child_vars)}])"

                elif op == CompositionOp.OR:
                    imports.add("from dreadnode.core.scorer import or_")
                    return f"{var_name} = or_([{', '.join(child_vars)}])"

                elif op == CompositionOp.SCALE:
                    imports.add("from dreadnode.core.scorer import scale")
                    factor = node.config.get("factor", 1.0)
                    return f"{var_name} = scale({child_vars[0]}, {factor})"

                elif op == CompositionOp.CLIP:
                    imports.add("from dreadnode.core.scorer import clip")
                    min_v = node.config.get("min_val", 0.0)
                    max_v = node.config.get("max_val", 1.0)
                    return f"{var_name} = clip({child_vars[0]}, {min_v}, {max_v})"

                elif op == CompositionOp.THRESHOLD:
                    imports.add("from dreadnode.core.scorer import threshold")
                    t = node.config.get("threshold", 0.5)
                    return f"{var_name} = threshold({child_vars[0]}, {t})"

                elif op == CompositionOp.INVERT:
                    imports.add("from dreadnode.core.scorer import invert")
                    return f"{var_name} = invert({child_vars[0]})"

                return f"# {var_name} = unknown_operation"

        # Generate the tree
        final_line = generate_node(self.scorers[name], name)
        code_lines.append(final_line)

        # Build final code
        import_str = "\n".join(sorted(imports))
        code_str = "\n".join(code_lines)

        return f'''"""
{name} Scorer

Auto-generated composite scorer.
"""

{import_str}

# Build scorer components
{code_str}

# Final scorer ready for use
scorer = {name}
'''


# Example usage
if __name__ == "__main__":
    composer = ScorerComposer()

    # Create primitives
    composer.create_primitive("length", "length", normalize_to=200)
    composer.create_primitive("has_summary", "contains", target="summary")
    composer.create_primitive("relevance", "similarity")

    # Compose
    composer.avg("content_quality", ["length", "relevance"])
    composer.all_pass("validation", ["has_summary", "content_quality"])
    composer.threshold("final", "validation", threshold=0.5)

    # Visualize
    print(composer.visualize("final"))

    # Generate code
    print(composer.generate_code("final"))
```

## API Reference

### ScorerComposer Methods

| Method                                   | Description                |
| ---------------------------------------- | -------------------------- |
| `create_primitive(name, type, **config)` | Create a primitive scorer  |
| `compose(name, op, children, **config)`  | Compose with any operation |
| `avg(name, children)`                    | Average scorers            |
| `weighted_avg(name, children, weights)`  | Weighted average           |
| `all_pass(name, children)`               | AND (all truthy)           |
| `any_pass(name, children)`               | OR (any truthy)            |
| `scale(name, child, factor)`             | Scale output               |
| `clip(name, child, min, max)`            | Clip to range              |
| `threshold(name, child, t)`              | Binary threshold           |
| `invert(name, child)`                    | 1 - value                  |
| `visualize(name)`                        | ASCII tree visualization   |
| `generate_code(name)`                    | Generate Python code       |

### Primitive Scorers

| Primitive   | Description         | Config                 |
| ----------- | ------------------- | ---------------------- |
| length      | Text length score   | normalize_to           |
| word_count  | Word count score    | normalize_to           |
| contains    | Text containment    | target, case_sensitive |
| regex_match | Regex matching      | pattern                |
| sentiment   | Sentiment score     | -                      |
| readability | Readability score   | metric                 |
| similarity  | Semantic similarity | model                  |
| llm_judge   | LLM evaluation      | model, rubric          |
| custom      | Custom function     | function               |

## Example: Building a Complex Scorer

```python
from scorer_composer import ScorerComposer

composer = ScorerComposer()

# Level 1: Primitives
composer.create_primitive("length_ok", "length", normalize_to=500)
composer.create_primitive("has_intro", "contains", target="introduction")
composer.create_primitive("has_conclusion", "contains", target="conclusion")
composer.create_primitive("semantic_match", "similarity")
composer.create_primitive("quality", "llm_judge", model="gpt-4")

# Level 2: Structure check
composer.all_pass("has_structure", ["has_intro", "has_conclusion"])

# Level 3: Content quality
composer.weighted_avg("content_score",
    ["semantic_match", "quality"],
    weights=[0.4, 0.6]
)

# Level 4: Overall score
composer.avg("overall", ["length_ok", "has_structure", "content_score"])

# Level 5: Final with threshold
composer.threshold("final_scorer", "overall", threshold=0.6)

# Visualize the tree
print(composer.visualize("final_scorer"))

# Generate code
code = composer.generate_code("final_scorer")
print(code)
```

Output:

```
Scorer Tree: final_scorer
└── final_scorer (threshold)
    └── overall (avg)
        ├── length_ok [length]
        ├── has_structure (and)
        │   ├── has_intro [contains]
        │   └── has_conclusion [contains]
        └── content_score (weighted_avg)
            ├── semantic_match [similarity]
            └── quality [llm_judge]
```
