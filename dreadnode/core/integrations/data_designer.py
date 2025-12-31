from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if t.TYPE_CHECKING:
    import pandas as pd

    from dreadnode.core.agents.trajectory import Trajectory
    from dreadnode.core.evaluations import DatasetLike


class SamplerType(str, Enum):
    """Statistical sampler types for non-LLM columns."""

    CATEGORY = "category"
    SUBCATEGORY = "subcategory"
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    BERNOULLI = "bernoulli"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    PERSON = "person"
    UUID = "uuid"


class SamplingStrategy(str, Enum):
    """How to sample from seed datasets."""

    ORDERED = "ordered"
    SHUFFLE = "shuffle"


class ColumnConfig(BaseModel, ABC):
    """Base class for all column configurations."""

    model_config = ConfigDict(use_enum_values=True)

    name: str
    """The name of the column in the generated dataset."""

    @abstractmethod
    def to_data_designer_config(self) -> dict[str, t.Any]:
        """Convert to Data Designer column configuration."""
        ...


class SamplerColumn(ColumnConfig):
    """
    Statistical sampler column for generating diverse seed values.

    These columns use probabilistic methods rather than LLMs, making them
    fast and deterministic. Use them to create the foundation that LLM
    columns can reference.

    Example:
        ```python
        SamplerColumn(
            name="difficulty",
            sampler_type=SamplerType.CATEGORY,
            values=["easy", "medium", "hard"],
            weights=[0.2, 0.5, 0.3],  # Optional probability weights
        )
        ```
    """

    sampler_type: SamplerType
    """The type of statistical sampler to use."""

    # Category sampler params
    values: list[str] | None = None
    """Values for category sampler."""
    weights: list[float] | None = None
    """Optional probability weights for category sampler."""

    # Numeric sampler params
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    std: float | None = None

    # DateTime params
    start_date: str | None = None
    end_date: str | None = None

    def to_data_designer_config(self) -> dict[str, t.Any]:
        config: dict[str, t.Any] = {
            "name": self.name,
            "sampler_type": self.sampler_type,
            "params": {},
        }

        if self.sampler_type == SamplerType.CATEGORY:
            config["params"]["values"] = self.values
            if self.weights:
                config["params"]["weights"] = self.weights

        elif self.sampler_type in (SamplerType.UNIFORM, SamplerType.GAUSSIAN):
            if self.min_value is not None:
                config["params"]["min"] = self.min_value
            if self.max_value is not None:
                config["params"]["max"] = self.max_value
            if self.mean is not None:
                config["params"]["mean"] = self.mean
            if self.std is not None:
                config["params"]["std"] = self.std

        elif self.sampler_type == SamplerType.DATETIME:
            if self.start_date:
                config["params"]["start"] = self.start_date
            if self.end_date:
                config["params"]["end"] = self.end_date

        return config


class LLMColumn(ColumnConfig):
    """
    LLM-generated column using prompt templates.

    Use Jinja2 syntax to reference other columns: {{ column_name }}

    Example:
        ```python
        LLMColumn(
            name="challenge_description",
            prompt=\"\"\"
                Create a {{ difficulty }} {{ category }} CTF challenge.
                The flag format should be: flag{...}
            \"\"\",
            model_alias="text",
        )
        ```
    """

    prompt: str
    """Jinja2 template for the generation prompt."""
    model_alias: str = "text"
    """Model alias defined in model configs."""

    def to_data_designer_config(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "type": "llm-text",
            "prompt": dedent(self.prompt).strip(),
            "model_alias": self.model_alias,
        }


class StructuredColumn(ColumnConfig):
    """
    LLM column that generates structured data matching a Pydantic schema.

    Example:
        ```python
        class Challenge(BaseModel):
            title: str
            description: str
            hints: list[str]
            flag: str

        StructuredColumn(
            name="challenge",
            prompt="Generate a {{ difficulty }} CTF challenge",
            output_schema=Challenge,
        )
        ```
    """

    prompt: str
    """Jinja2 template for the generation prompt."""
    output_schema: type[BaseModel]
    """Pydantic model defining the output structure."""
    model_alias: str = "structured"
    """Model alias (should support structured outputs)."""

    def to_data_designer_config(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "type": "llm-structured",
            "prompt": dedent(self.prompt).strip(),
            "model_alias": self.model_alias,
            "output_format": self.output_schema,
        }


class JudgeColumn(ColumnConfig):
    """
    LLM-as-judge column for scoring generated content.

    Example:
        ```python
        JudgeColumn(
            name="quality_score",
            prompt="Rate the quality of this challenge: {{ challenge }}",
            target_column="challenge_description",
            rubrics=["clarity", "difficulty_appropriate", "solvable"],
        )
        ```
    """

    prompt: str
    """Evaluation prompt template."""
    target_column: str
    """Column to evaluate."""
    rubrics: list[str] | None = None
    """Evaluation criteria."""
    model_alias: str = "judge"

    def to_data_designer_config(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "type": "llm-judge",
            "prompt": dedent(self.prompt).strip(),
            "target_column": self.target_column,
            "rubrics": self.rubrics,
            "model_alias": self.model_alias,
        }


class SeedColumn(ColumnConfig):
    """
    Column that references data from a seed dataset.

    Use this to pull values directly from your existing data.

    Example:
        ```python
        SeedColumn(name="original_prompt", seed_column="user_message")
        ```
    """

    seed_column: str
    """Name of the column in the seed dataset to reference."""

    def to_data_designer_config(self) -> dict[str, t.Any]:
        return {
            "name": self.name,
            "type": "seed-reference",
            "seed_column": self.seed_column,
        }


# =============================================================================
# Model Configuration
# =============================================================================


class InferenceParams(BaseModel):
    """Parameters for LLM inference."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048


class ModelConfig(BaseModel):
    """Configuration for an LLM model used in generation."""

    alias: str
    """Alias to reference this model in columns."""
    model: str
    """Model identifier (e.g., 'meta/llama-3.3-70b-instruct')."""
    provider: str = "nvidiabuild"
    """Model provider."""
    inference_params: InferenceParams = Field(default_factory=InferenceParams)


class DatasetSchema(BaseModel):
    """
    Complete schema definition for synthetic dataset generation.

    Example:
        ```python
        schema = DatasetSchema(
            columns=[
                SamplerColumn(name="difficulty", sampler_type=SamplerType.CATEGORY, values=["easy", "medium", "hard"]),
                SamplerColumn(name="category", sampler_type=SamplerType.CATEGORY, values=["web", "crypto", "pwn"]),
                LLMColumn(
                    name="challenge",
                    prompt="Create a {{ difficulty }} {{ category }} CTF challenge with a flag.",
                ),
            ],
            models=[
                ModelConfig(alias="text", model="meta/llama-3.3-70b-instruct"),
            ],
        )
        ```
    """

    columns: list[ColumnConfig]
    """Column definitions in generation order."""
    models: list[ModelConfig] = Field(default_factory=list)
    """Model configurations."""

    # Seed data configuration
    seed_data: DatasetLike | Path | str | None = None
    """Optional seed dataset to bootstrap generation."""
    seed_sampling_strategy: SamplingStrategy = SamplingStrategy.SHUFFLE
    """How to sample from seed data."""

    def to_data_designer_config(self) -> dict[str, t.Any]:
        """Convert to Data Designer configuration format."""
        return {
            "model_configs": [
                {
                    "alias": m.alias,
                    "model": m.model,
                    "provider": m.provider,
                    "inference_parameters": m.inference_params.model_dump(),
                }
                for m in self.models
            ],
            "columns": [col.to_data_designer_config() for col in self.columns],
        }


class DatasetGenerator(BaseModel):
    """
    Generate synthetic datasets using NVIDIA Data Designer.

    This component integrates Data Designer with the Dreadnode SDK,
    providing a unified interface for creating evaluation datasets,
    training data, and test scenarios.

    Example - Basic Usage:
        ```python
        generator = DatasetGenerator(
            base_url="http://localhost:8080",  # Or use NVIDIA API
        )

        dataset = await generator.generate(
            schema=DatasetSchema(
                columns=[
                    SamplerColumn(name="topic", sampler_type=SamplerType.CATEGORY, values=["AI", "Security", "Web"]),
                    LLMColumn(name="question", prompt="Generate a question about {{ topic }}"),
                ],
                models=[ModelConfig(alias="text", model="meta/llama-3.3-70b-instruct")],
            ),
            num_records=100,
        )

        # Use directly with agent evaluation
        result = await agent.evaluate(dataset=dataset, scorers=[accuracy])
        ```

    Example - From Pydantic Schema:
        ```python
        class CTFChallenge(BaseModel):
            goal: str
            category: str
            difficulty: str
            hints: list[str]
            flag: str

        dataset = await generator.from_schema(
            output_type=CTFChallenge,
            num_records=50,
            instructions="Generate realistic CTF challenges",
            samplers={
                "category": ["web", "crypto", "pwn", "forensics"],
                "difficulty": ["easy", "medium", "hard"],
            },
        )
        ```

    Example - Seeding from Existing Data:
        ```python
        # Use existing trajectories to generate similar scenarios
        seed_data = [t.trajectory_to_nemo() for t in successful_trajectories]

        dataset = await generator.generate(
            schema=DatasetSchema(
                seed_data=seed_data,
                columns=[
                    SeedColumn(name="original_goal", seed_column="goal"),
                    LLMColumn(
                        name="goal",
                        prompt="Create a variation of: {{ original_goal }}",
                    ),
                ],
            ),
            num_records=200,
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str | None = None
    """Base URL for Data Designer service. If None, uses NVIDIA API."""
    api_key: str | None = None
    """API key for NVIDIA API (uses NVIDIA_API_KEY env var if not set)."""

    default_model: str = "meta/llama-3.3-70b-instruct"
    """Default model for LLM columns."""
    default_provider: str = "nvidiabuild"
    """Default model provider."""

    # Private client instance
    _client: t.Any = PrivateAttr(None)

    def _get_client(self) -> t.Any:
        """Get or create the Data Designer client."""
        if self._client is not None:
            return self._client

        try:
            # Try the standalone data-designer package first
            from data_designer import DataDesigner

            self._client = DataDesigner(api_key=self.api_key)
        except ImportError:
            try:
                # Fall back to NeMo microservices SDK
                from nemo_microservices.data_designer.essentials import (
                    NeMoDataDesignerClient,
                )

                if self.base_url:
                    self._client = NeMoDataDesignerClient(base_url=self.base_url)
                else:
                    import os

                    from nemo_microservices import NeMoMicroservices

                    self._client = NeMoDataDesignerClient(
                        client=NeMoMicroservices(
                            base_url=os.environ.get(
                                "NEMO_MICROSERVICES_BASE_URL",
                                "https://integrate.api.nvidia.com/v1",
                            )
                        )
                    )
            except ImportError as e:
                raise ImportError(
                    "Neither 'data-designer' nor 'nemo-microservices[data-designer]' is installed. "
                    "Install with: pip install data-designer OR pip install nemo-microservices[data-designer]"
                ) from e

        return self._client

    async def generate(
        self,
        schema: DatasetSchema,
        num_records: int = 100,
        *,
        preview: bool = False,
        output_format: t.Literal["list", "dataframe"] = "list",
    ) -> list[dict[str, t.Any]] | pd.DataFrame:
        """
        Generate a synthetic dataset from a schema definition.

        Args:
            schema: The dataset schema defining columns and models.
            num_records: Number of records to generate.
            preview: If True, generate a small preview (typically 5-10 records).
            output_format: Return as list of dicts or pandas DataFrame.

        Returns:
            Generated dataset in the requested format.
        """
        import asyncio

        client = self._get_client()

        # Build configuration
        try:
            from data_designer import DataDesignerConfigBuilder
        except ImportError:
            from nemo_microservices.data_designer.essentials import (
                DataDesignerConfigBuilder,
            )

        config = schema.to_data_designer_config()
        builder = DataDesignerConfigBuilder(model_configs=config["model_configs"])

        # Add columns
        for col_config in config["columns"]:
            builder.add_column(col_config)

        # Handle seed data if provided
        if schema.seed_data is not None:
            await self._configure_seed_data(builder, schema)

        # Generate data
        if preview:
            result = await asyncio.to_thread(client.preview, builder)
            records = result.dataset.to_dict("records")
        else:
            job_result = await asyncio.to_thread(
                client.create, builder, num_records=num_records, wait_until_done=True
            )
            dataset = job_result.load_dataset()
            records = dataset.to_dict("records")

        if output_format == "dataframe":
            import pandas as pd

            return pd.DataFrame(records)

        return records

    async def from_schema(
        self,
        output_type: type[BaseModel],
        num_records: int = 100,
        *,
        instructions: str | None = None,
        samplers: dict[str, list[str] | SamplerColumn] | None = None,
        model: str | None = None,
    ) -> list[dict[str, t.Any]]:
        """
        Generate a dataset from a Pydantic model definition.

        This is a convenience method that automatically builds a schema
        from a Pydantic model, using field names and types to infer
        column configurations.

        Args:
            output_type: Pydantic model defining the output structure.
            num_records: Number of records to generate.
            instructions: Optional instructions for generation.
            samplers: Dict mapping field names to category values or SamplerColumn configs.
            model: Model to use for generation.

        Returns:
            List of dicts matching the output_type schema.

        Example:
            ```python
            class TestCase(BaseModel):
                input: str
                expected_output: str
                difficulty: str

            cases = await generator.from_schema(
                output_type=TestCase,
                samplers={"difficulty": ["easy", "medium", "hard"]},
                instructions="Generate test cases for a calculator",
            )
            ```
        """
        columns: list[ColumnConfig] = []
        sampler_refs: list[str] = []

        # Process samplers first (they become seed columns for LLM generation)
        if samplers:
            for field_name, sampler_config in samplers.items():
                if isinstance(sampler_config, SamplerColumn):
                    columns.append(sampler_config)
                elif isinstance(sampler_config, list):
                    columns.append(
                        SamplerColumn(
                            name=field_name,
                            sampler_type=SamplerType.CATEGORY,
                            values=sampler_config,
                        )
                    )
                sampler_refs.append(field_name)

        # Build the prompt for remaining fields
        fields_info = output_type.model_json_schema()
        field_descriptions = []
        for name, field_schema in fields_info.get("properties", {}).items():
            if name not in sampler_refs:
                field_type = field_schema.get("type", "string")
                desc = field_schema.get("description", "")
                field_descriptions.append(f"- {name} ({field_type}): {desc}")

        # Build context from samplers
        sampler_context = ""
        if sampler_refs:
            sampler_context = "Given:\n" + "\n".join(
                f"- {ref}: {{{{ {ref} }}}}" for ref in sampler_refs
            )

        prompt = f"""
        {instructions or "Generate the following data:"}
        
        {sampler_context}
        
        Generate a JSON object with these fields:
        {chr(10).join(field_descriptions)}
        """

        # Add structured column for the full output
        columns.append(
            StructuredColumn(
                name="generated",
                prompt=prompt,
                output_schema=output_type,
                model_alias="structured",
            )
        )

        schema = DatasetSchema(
            columns=columns,
            models=[
                ModelConfig(
                    alias="structured",
                    model=model or self.default_model,
                    provider=self.default_provider,
                )
            ],
        )

        results = await self.generate(schema, num_records)

        # Flatten the structured output
        flattened = []
        for record in results:
            flat_record = {k: v for k, v in record.items() if k != "generated"}
            if "generated" in record and isinstance(record["generated"], dict):
                flat_record.update(record["generated"])
            flattened.append(flat_record)

        return flattened

    async def from_trajectories(
        self,
        trajectories: list[Trajectory],
        num_records: int = 100,
        *,
        variation_prompt: str | None = None,
        include_successful_only: bool = True,
    ) -> list[dict[str, t.Any]]:
        """
        Generate new scenarios based on existing agent trajectories.

        This is useful for creating more diverse evaluation datasets
        from successful agent runs.

        Args:
            trajectories: List of trajectories to use as seed data.
            num_records: Number of new records to generate.
            variation_prompt: Custom prompt for generating variations.
            include_successful_only: Filter to only successful trajectories.

        Returns:
            New dataset based on trajectory patterns.
        """
        from dreadnode.core.agents.events import AgentEnd

        # Extract goal information from trajectories
        seed_data = []
        for traj in trajectories:
            # Check if successful
            if include_successful_only:
                end_events = traj.get_events_by_type(AgentEnd)
                if not end_events or end_events[-1].stop_reason != "finished":
                    continue

            # Extract the goal from the first user message
            goal = None
            for msg in traj.messages:
                if msg.role == "user":
                    goal = msg.content
                    break

            if goal:
                seed_data.append(
                    {
                        "original_goal": goal,
                        "num_steps": len(traj.steps),
                        "total_tokens": traj.usage.total_tokens,
                    }
                )

        if not seed_data:
            raise ValueError("No valid trajectories found to seed from")

        default_variation_prompt = """
        Based on this original goal: {{ original_goal }}

        Generate a new, similar but distinct goal that:
        - Tests similar capabilities
        - Has comparable complexity (original took {{ num_steps }} steps)
        - Is different enough to be a unique test case
        """

        schema = DatasetSchema(
            seed_data=seed_data,
            seed_sampling_strategy=SamplingStrategy.SHUFFLE,
            columns=[
                SeedColumn(name="original_goal", seed_column="original_goal"),
                SeedColumn(name="num_steps", seed_column="num_steps"),
                LLMColumn(
                    name="goal",
                    prompt=variation_prompt or default_variation_prompt,
                    model_alias="text",
                ),
            ],
            models=[
                ModelConfig(alias="text", model=self.default_model),
            ],
        )

        return await self.generate(schema, num_records)

    async def _configure_seed_data(
        self,
        builder: t.Any,
        schema: DatasetSchema,
    ) -> None:
        """Configure seed data on the builder."""
        import asyncio

        seed_data = schema.seed_data

        # Convert to appropriate format
        if isinstance(seed_data, (str, Path)):
            # File path - will be uploaded
            client = self._get_client()
            if hasattr(client, "upload_seed_dataset"):
                ref = await asyncio.to_thread(
                    client.upload_seed_dataset,
                    dataset=str(seed_data),
                    repo_id="dreadnode/seed-data",
                )
                builder.with_seed_dataset(
                    dataset_reference=ref,
                    sampling_strategy=schema.seed_sampling_strategy.value,
                )
        elif isinstance(seed_data, list):
            # List of dicts - convert to DataFrame and upload
            import pandas as pd

            df = pd.DataFrame(seed_data)

            client = self._get_client()
            if hasattr(client, "upload_seed_dataset"):
                ref = await asyncio.to_thread(
                    client.upload_seed_dataset,
                    dataset=df,
                    repo_id="dreadnode/seed-data",
                )
                builder.with_seed_dataset(
                    dataset_reference=ref,
                    sampling_strategy=schema.seed_sampling_strategy.value,
                )


def dataset_generator(
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: t.Any,
) -> DatasetGenerator:
    """
    Create a DatasetGenerator instance.

    Args:
        base_url: Data Designer service URL (None for NVIDIA API).
        api_key: API key for NVIDIA API.
        **kwargs: Additional configuration.

    Returns:
        Configured DatasetGenerator.

    Example:
        ```python
        gen = dataset_generator()  # Uses NVIDIA_API_KEY env var
        data = await gen.generate(schema, num_records=100)
        ```
    """
    return DatasetGenerator(base_url=base_url, api_key=api_key, **kwargs)


class GeneratedDataset(BaseModel):
    """
    A generated dataset that can be used with Dreadnode's evaluation system.

    This class wraps a generated dataset and provides methods for
    integration with Agent.evaluate(), Evaluation, and Study.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: list[dict[str, t.Any]]
    """The generated records."""
    schema: DatasetSchema | None = None
    """The schema used to generate the data."""
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    """Generation metadata."""

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> t.Iterator[dict[str, t.Any]]:
        return iter(self.records)

    def __getitem__(self, index: int) -> dict[str, t.Any]:
        return self.records[index]

    def to_list(self) -> list[dict[str, t.Any]]:
        """Return as list of dicts (compatible with DatasetLike)."""
        return self.records

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def to_parquet(self, path: str | Path) -> Path:
        """Save to Parquet file."""
        path = Path(path)
        self.to_dataframe().to_parquet(path)
        return path

    def to_jsonl(self, path: str | Path) -> Path:
        """Save to JSONL file."""
        import json

        path = Path(path)
        with open(path, "w") as f:
            f.writelines(json.dumps(record) + "\n" for record in self.records)
        return path

    def sample(self, n: int, seed: int | None = None) -> GeneratedDataset:
        """Return a random sample of records."""
        import random

        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.records, min(n, len(self.records)))
        return GeneratedDataset(
            records=sampled,
            schema=self.schema,
            metadata={**self.metadata, "sampled_from": len(self.records)},
        )

    def filter(self, predicate: t.Callable[[dict[str, t.Any]], bool]) -> GeneratedDataset:
        """Filter records by a predicate function."""
        filtered = [r for r in self.records if predicate(r)]
        return GeneratedDataset(
            records=filtered,
            schema=self.schema,
            metadata={**self.metadata, "filtered_from": len(self.records)},
        )


__all__ = [
    "ColumnConfig",
    "DatasetGenerator",
    "DatasetSchema",
    "GeneratedDataset",
    "InferenceParams",
    "JudgeColumn",
    "LLMColumn",
    "ModelConfig",
    "SamplerColumn",
    "SamplerType",
    "SamplingStrategy",
    "SeedColumn",
    "StructuredColumn",
    "dataset_generator",
]
