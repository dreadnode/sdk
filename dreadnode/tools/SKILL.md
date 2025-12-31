# Tool Selector Skill v2

An intelligent tool selection system that analyzes tasks and recommends appropriate Dreadnode tools based on semantic understanding and task requirements.

## Overview

The Tool Selector v2 provides advanced tool recommendation with:

- Semantic task analysis
- Tool compatibility checking
- Workflow optimization
- Tool combination suggestions

## Usage

```python
import typing as t
from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(Enum):
    """Categories of Dreadnode tools."""
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    PLANNING = "planning"
    EXECUTION = "execution"
    REPORTING = "reporting"
    SCORING = "scoring"
    TRANSFORM = "transform"
    ATTACK = "attack"
    SEARCH = "search"


@dataclass
class ToolInfo:
    """Information about a tool."""
    name: str
    category: ToolCategory
    description: str
    methods: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)


# Tool Registry
TOOL_REGISTRY: dict[str, ToolInfo] = {
    # Filesystem Tools
    "Filesystem": ToolInfo(
        name="Filesystem",
        category=ToolCategory.FILESYSTEM,
        description="Local and S3 file system operations",
        methods=["ls", "read_file", "write_file", "glob", "grep", "mkdir", "cp", "mv", "delete"],
        keywords=["file", "read", "write", "directory", "path", "storage", "s3", "upload", "download", "folder"],
        provides=["file_access", "storage"]
    ),

    # Memory Tools
    "Memory": ToolInfo(
        name="Memory",
        category=ToolCategory.MEMORY,
        description="Key-value memory for maintaining state",
        methods=["store", "recall", "forget", "list_keys"],
        keywords=["remember", "store", "recall", "context", "history", "state", "cache", "persist"],
        provides=["state_management", "context"]
    ),

    # Planning Tools
    "Planning": ToolInfo(
        name="Planning",
        category=ToolCategory.PLANNING,
        description="Task planning and organization",
        methods=["create_plan", "update_plan", "complete_step", "get_next_step"],
        keywords=["plan", "strategy", "steps", "organize", "breakdown", "task", "schedule", "goal"],
        provides=["task_planning", "organization"]
    ),

    # Execution Tools
    "Execute": ToolInfo(
        name="Execute",
        category=ToolCategory.EXECUTION,
        description="Code and command execution in sandbox",
        methods=["run_code", "run_command", "run_script"],
        keywords=["run", "execute", "code", "script", "command", "shell", "python", "bash"],
        requires=["sandbox_environment"],
        provides=["code_execution"]
    ),

    # Reporting Tools
    "Reporting": ToolInfo(
        name="Reporting",
        category=ToolCategory.REPORTING,
        description="Generate reports and summaries",
        methods=["create_report", "add_section", "add_finding", "export"],
        keywords=["report", "output", "summary", "results", "document", "export", "findings"],
        provides=["documentation", "output_generation"]
    ),

    # Scorers
    "contains": ToolInfo(
        name="contains",
        category=ToolCategory.SCORING,
        description="Check if output contains specific text",
        methods=["score"],
        keywords=["check", "contains", "match", "text", "find", "search"],
        provides=["text_matching"]
    ),
    "similarity": ToolInfo(
        name="similarity",
        category=ToolCategory.SCORING,
        description="Semantic similarity scoring",
        methods=["score"],
        keywords=["similar", "semantic", "embedding", "compare", "relevance"],
        requires=["embeddings_model"],
        provides=["semantic_matching"]
    ),
    "llm_judge": ToolInfo(
        name="llm_judge",
        category=ToolCategory.SCORING,
        description="LLM-based evaluation",
        methods=["score"],
        keywords=["judge", "evaluate", "llm", "quality", "assess", "grade"],
        requires=["llm_model"],
        provides=["llm_evaluation"]
    ),

    # Transforms
    "encoding": ToolInfo(
        name="encoding",
        category=ToolCategory.TRANSFORM,
        description="Text encoding transforms (base64, rot13, etc.)",
        methods=["apply"],
        keywords=["encode", "decode", "base64", "rot13", "transform"],
        provides=["text_encoding"]
    ),
    "cipher": ToolInfo(
        name="cipher",
        category=ToolCategory.TRANSFORM,
        description="Cipher transforms for obfuscation",
        methods=["apply"],
        keywords=["cipher", "encrypt", "obfuscate", "caesar", "vigenere"],
        provides=["text_obfuscation"]
    ),

    # Attack Tools (AIRT)
    "tap": ToolInfo(
        name="tap",
        category=ToolCategory.ATTACK,
        description="Tree of Attacks with Pruning",
        methods=["attack"],
        keywords=["attack", "red team", "adversarial", "jailbreak", "prompt injection"],
        requires=["llm_model"],
        provides=["adversarial_testing"]
    ),
    "goat": ToolInfo(
        name="goat",
        category=ToolCategory.ATTACK,
        description="Generative Offensive Agent Tester",
        methods=["attack"],
        keywords=["attack", "agent", "offensive", "test", "security"],
        requires=["llm_model"],
        provides=["agent_testing"]
    ),

    # Search Strategies
    "grid": ToolInfo(
        name="grid",
        category=ToolCategory.SEARCH,
        description="Grid search over parameter space",
        methods=["search"],
        keywords=["grid", "exhaustive", "combinations", "parameters"],
        provides=["hyperparameter_search"]
    ),
    "optuna": ToolInfo(
        name="optuna",
        category=ToolCategory.SEARCH,
        description="Bayesian optimization search",
        methods=["search"],
        keywords=["bayesian", "optimization", "optuna", "smart", "efficient"],
        provides=["intelligent_search"]
    ),
}


class ToolSelector:
    """
    Intelligent tool selection based on task analysis.
    """

    def __init__(self, registry: dict[str, ToolInfo] = None):
        self.registry = registry or TOOL_REGISTRY

    def analyze_task(self, task_description: str) -> dict:
        """
        Analyze a task and return recommended tools.

        Args:
            task_description: Natural language description of the task

        Returns:
            Analysis with tool recommendations
        """
        task_lower = task_description.lower()

        # Score each tool
        tool_scores: dict[str, float] = {}
        for name, info in self.registry.items():
            score = 0
            for keyword in info.keywords:
                if keyword in task_lower:
                    score += 1
            if score > 0:
                tool_scores[name] = score

        # Sort by score
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)

        # Group by category
        categories: dict[str, list[dict]] = {}
        for name, score in sorted_tools:
            info = self.registry[name]
            cat = info.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "name": name,
                "score": score,
                "description": info.description,
                "methods": info.methods
            })

        return {
            "task": task_description,
            "recommended_tools": sorted_tools[:5],
            "by_category": categories,
            "total_matches": len(sorted_tools)
        }

    def get_tool_info(self, tool_name: str) -> dict:
        """Get detailed information about a tool."""
        if tool_name not in self.registry:
            return {"error": f"Tool '{tool_name}' not found"}

        info = self.registry[tool_name]
        return {
            "name": info.name,
            "category": info.category.value,
            "description": info.description,
            "methods": info.methods,
            "keywords": info.keywords,
            "requires": info.requires,
            "provides": info.provides
        }

    def suggest_workflow(self, primary_task: str, secondary_tasks: list[str] = None) -> dict:
        """
        Suggest an optimal tool workflow for complex tasks.

        Args:
            primary_task: Main task description
            secondary_tasks: Supporting task descriptions

        Returns:
            Workflow with ordered tool suggestions
        """
        primary = self.analyze_task(primary_task)

        workflow = {
            "primary_tools": primary["recommended_tools"],
            "secondary_tools": {},
            "workflow_order": []
        }

        all_tools = set(t[0] for t in primary["recommended_tools"])

        if secondary_tasks:
            for task in secondary_tasks:
                analysis = self.analyze_task(task)
                workflow["secondary_tools"][task] = analysis["recommended_tools"]
                all_tools.update(t[0] for t in analysis["recommended_tools"])

        # Determine workflow order based on categories
        category_order = [
            ToolCategory.PLANNING,
            ToolCategory.FILESYSTEM,
            ToolCategory.MEMORY,
            ToolCategory.EXECUTION,
            ToolCategory.TRANSFORM,
            ToolCategory.SCORING,
            ToolCategory.ATTACK,
            ToolCategory.SEARCH,
            ToolCategory.REPORTING
        ]

        ordered_tools = []
        for cat in category_order:
            for tool in all_tools:
                if tool in self.registry and self.registry[tool].category == cat:
                    if tool not in ordered_tools:
                        ordered_tools.append(tool)

        workflow["workflow_order"] = ordered_tools

        return workflow

    def check_compatibility(self, tools: list[str]) -> dict:
        """
        Check if a set of tools are compatible.

        Args:
            tools: List of tool names

        Returns:
            Compatibility analysis
        """
        all_requires = set()
        all_provides = set()
        missing = []

        for tool in tools:
            if tool not in self.registry:
                continue
            info = self.registry[tool]
            all_requires.update(info.requires)
            all_provides.update(info.provides)

        # Check if requirements are met
        for req in all_requires:
            if req not in all_provides:
                missing.append(req)

        return {
            "tools": tools,
            "compatible": len(missing) == 0,
            "requires": list(all_requires),
            "provides": list(all_provides),
            "missing_requirements": missing
        }


# Convenience function
def select_tools(task: str) -> dict:
    """Quick tool selection for a task."""
    selector = ToolSelector()
    return selector.analyze_task(task)


# Example usage
if __name__ == "__main__":
    selector = ToolSelector()

    # Analyze a task
    result = selector.analyze_task(
        "I need to read files from S3, analyze their content, and generate a report"
    )
    print("Task Analysis:")
    print(f"  Recommended: {result['recommended_tools']}")

    # Get tool info
    info = selector.get_tool_info("Filesystem")
    print(f"\nFilesystem Tool: {info['description']}")

    # Suggest workflow
    workflow = selector.suggest_workflow(
        "Evaluate an LLM on a dataset",
        ["Score responses for quality", "Save results to file"]
    )
    print(f"\nWorkflow Order: {workflow['workflow_order']}")
```

## API Reference

### ToolSelector

| Method                                 | Description                             |
| -------------------------------------- | --------------------------------------- |
| `analyze_task(description)`            | Analyze a task and recommend tools      |
| `get_tool_info(name)`                  | Get detailed info about a specific tool |
| `suggest_workflow(primary, secondary)` | Suggest optimal tool ordering           |
| `check_compatibility(tools)`           | Verify tool set compatibility           |

### Tool Categories

| Category   | Description         | Example Tools                   |
| ---------- | ------------------- | ------------------------------- |
| FILESYSTEM | File operations     | Filesystem, S3Filesystem        |
| MEMORY     | State management    | Memory                          |
| PLANNING   | Task organization   | Planning                        |
| EXECUTION  | Code running        | Execute                         |
| REPORTING  | Output generation   | Reporting                       |
| SCORING    | Evaluation          | contains, similarity, llm_judge |
| TRANSFORM  | Data transformation | encoding, cipher                |
| ATTACK     | Adversarial testing | tap, goat                       |
| SEARCH     | Optimization        | grid, optuna                    |

## Integration Example

```python
from tool_selector_v2 import ToolSelector, select_tools

# Quick selection
recommendations = select_tools("Process CSV files and generate statistics")

# Full workflow planning
selector = ToolSelector()
workflow = selector.suggest_workflow(
    "Build an evaluation pipeline",
    ["Load test data", "Score outputs", "Generate report"]
)

# Use recommended tools in agent
from dreadnode import Agent

agent = Agent(
    name="workflow-agent",
    tools=[get_tool(name) for name, _ in workflow["recommended_tools"][:3]]
)
```
