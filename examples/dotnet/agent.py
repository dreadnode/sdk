import random
import typing as t
from dataclasses import dataclass
from pathlib import Path

import cyclopts
import yaml

import dreadnode
from dreadnode.agent.agent import TaskAgent
from dreadnode.agent.tools import (
    DotnetReversing,
)

app = cyclopts.App()


@cyclopts.Parameter(name="*", group="args")
@dataclass
class Args:
    model: str
    """Model to use for inference"""
    path: str
    """Directory of binaries to analyze or other supported identifier"""
    nuget: bool = False
    """Treat the path as a NuGet package id or path to a list of packages"""
    task: str = "Find only critical vulnerabilities"
    """Task presented to the agent"""
    max_steps: int = 25
    """Maximum number of iterations per agent"""
    concurrency: int = 3
    """Maximum number of agents to run in parallel at any given time"""
    log_level: str = "INFO"
    """Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""


@cyclopts.Parameter(name="*", group="dreadnode")
@dataclass
class DreadnodeArgs:
    server: str | None = None
    """Dreadnode server URL"""
    token: str | None = None
    """Dreadnode API token"""
    project: str = "dotnet-reversing-final"
    """Project name"""
    console: t.Annotated[bool, cyclopts.Parameter(negative=False)] = False
    """Show span information in the console"""


def load_instructions(instructions_file: Path) -> str:
    """
    Load the instructions from a YAML file.
    """
    with Path.open(instructions_file) as file:
        instructions = yaml.safe_load(file)
        return instructions.get("instructions", "")


@app.default
async def main(*, args: Args, dn_args: DreadnodeArgs | None = None) -> None:
    dn_args = dn_args or DreadnodeArgs()
    dreadnode.configure(
        server=dn_args.server,
        token=dn_args.token,
        project=dn_args.project,
        console=dn_args.console,
    )

    models = [
        "gpt-4o",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        "gpt-4.1",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/moonshotai/kimi-k2-instruct",
        "o4-mini-2025-04-16",
    ]

    reversing_agent = TaskAgent(
        name="dotnet-reversing",
        description="A basic agent that can handle .NET reversing tasks.",
        instructions=load_instructions(Path("examples/dotnet/prompts/reversing_agent.yml")),
        model=random.choice(models),
        max_steps=args.max_steps,
        tools=DotnetReversing,
    )

    verification_agent = TaskAgent(
        name="verify_findings",
        description="A basic agent that analyzes findings and generates exploit instructions.",
        instructions=load_instructions(Path("examples/dotnet/prompts/verification_agent.yml")),
        model=random.choice(models),
        max_steps=args.max_steps,
        tools=[],
    )

    rev_results = await reversing_agent.run()
    ver_results = await verification_agent.run(rev_results)

    return ver_results
