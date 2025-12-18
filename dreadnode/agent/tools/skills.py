import re
import typing as t
from pathlib import Path

import yaml
from loguru import logger
from pydantic import Field, model_validator

from dreadnode.agent.tools.base import Toolset, tool_method
from dreadnode.meta import Config, Model


class Skill(Model):
    """
    A skill represents a set of instructions or examples that an agent can use.
    """

    name: str = ""
    """The name of the skill, derived from the frontmatter or filename."""
    description: str = ""
    """A brief description of the skill, derived from the frontmatter or content."""
    content: str
    """The content of the skill."""
    path: Path | None = Field(default=None, exclude=True)
    """The path to the skill file."""

    @model_validator(mode="before")
    @classmethod
    def parse_frontmatter(cls, data: t.Any) -> t.Any:
        if not isinstance(data, dict) or "content" not in data:
            return data

        content = data["content"]
        match = re.match(r"^---\s*\n(.*?)\n---(?:\s*\n|$)", content, re.DOTALL)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                if isinstance(frontmatter, dict):
                    data["name"] = frontmatter.get("name") or data.get("name")
                    data["description"] = frontmatter.get("description") or data.get("description")
                    data["content"] = content[match.end() :].strip()
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse frontmatter in skill: {e}")

        if not data.get("name") and data.get("path"):
            data["name"] = Path(data["path"]).stem

        if not data.get("description"):
            for line in data["content"].strip().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                data["description"] = stripped
                break

        if not data.get("description"):
            data["description"] = "No description available."

        return data


class Skills(Toolset):
    """Tools for interacting with agent skills."""

    skills_dir: str = Config("skills")

    @tool_method
    def view_skill(self, name: str) -> str:
        """
        View the detailed content of a skill by name.

        Args:
            name: The name of the skill to view.
        """
        skills = self.load(self.skills_dir)
        skill = next((s for s in skills if s.name == name), None)

        if skill:
            return f"# Skill: {skill.name}\n\n{skill.content}"

        return f"Skill '{name}' not found."

    @staticmethod
    def load(directory: str | Path) -> list[Skill]:
        """
        Load skills from a directory.
        """
        skills_path = Path(directory)
        if not skills_path.exists() or not skills_path.is_dir():
            return []

        skills = []
        for file in skills_path.glob("*"):
            if file.is_file() and not file.name.startswith("."):
                try:
                    skills.append(
                        Skill(
                            content=file.read_text(encoding="utf-8"),
                            path=file,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.opt(exception=True).warning(f"Failed to load skill from {file}: {e}")
                    continue

        return sorted(skills, key=lambda s: s.name)
