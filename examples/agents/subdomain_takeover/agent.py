import typing as t
from pathlib import Path
import re

from rich.console import Console
from cyclopts import App

from dreadnode.agent.agent import Agent
from dreadnode.agent.tools.bbot.tool import BBotTool
from dreadnode.agent.tools.kali.tool import KaliTool

# Import necessary components for Pydantic dataclass fix
from dreadnode.agent.events import (
    Event, AgentStart, StepStart, GenerationEnd, 
    AgentStalled, AgentError, ToolStart, ToolEnd, AgentEnd
)
from dreadnode.agent.state import State
from dreadnode.agent.result import AgentResult
from dreadnode.agent.reactions import Reaction

import pydantic.dataclasses
try:
    critical_classes = [Event, AgentStart, StepStart, GenerationEnd, 
                       AgentStalled, AgentError, ToolStart, ToolEnd, AgentEnd]
    
    for event_class in critical_classes:
        pydantic.dataclasses.rebuild_dataclass(event_class)
except Exception:
    pass

"""Usage:
uv run python examples/agents/subdomain_takeover/agent.py validate test.example.com
uv run python examples/agents/subdomain_takeover/agent.py hunt --targets "/path/to/dir/subdomains.txt"
"""

console = Console()
app = App()


def create_takeover_agent() -> Agent:
    """Create a subdomain takeover analysis agent."""
    return Agent(
        name="subdomain-takeover-agent",
        description="An agent that analyzes subdomains for takeover vulnerabilities",
        model="gpt-4",
        tools=[BBotTool(), KaliTool()],
        instructions="""You are an expert at detecting subdomain takeover vulnerabilities.

FOCUS: Look for subdomains with DNS records (CNAME/A) pointing to unclaimed third-party services.

Key patterns:
- DNS resolves to third-party service (AWS S3, GitHub Pages, Heroku, Azure, Shopify, etc.)
- Service responds with error messages indicating unclaimed/deleted resource:
  * "No such bucket" 
  * "This site isn't configured"
  * "Project not found"
  * "There isn't a GitHub Pages site here"
  * "herokucdn.com/error-pages"

Example vulnerability:
marketing.example.com → CNAME → myapp.herokudns.com (but myapp is deleted/unclaimed)

Report ONLY actual takeover vulnerabilities, not general DNS misconfigurations."""
    )


def display_analysis_result(result: AgentResult, subdomain: str) -> None:
    """Display the agent's analysis result."""
    if not result or not result.messages:
        console.print("Analysis completed but no content available")
        return
    
    # Show which tools the agent decided to use
    tools_used = []
    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tools_used.append(tool_call.function.name)
    
    if tools_used:
        console.print(f"Agent used: {', '.join(tools_used)}")
    
    final_message = result.messages[-1]
    if final_message.content:
        console.print(f"\nAnalysis for {subdomain}:")
        console.print(final_message.content)
        console.print(f"\nProcessed {len(result.messages)} messages in {result.steps} steps")


@app.command
async def modules() -> None:
    """List available BBOT modules."""
    tool = await BBotTool.create()
    tool.get_modules()


@app.command
async def presets() -> None:
    """List available BBOT presets.""" 
    tool = await BBotTool.create()
    tool.get_presets()


@app.command
async def flags() -> None:
    """List available BBOT flags."""
    tool = await BBotTool.create()
    tool.get_flags()


@app.command
async def events() -> None:
    """List available BBOT event types."""
    tool = await BBotTool.create()
    tool.get_events()


@app.command
async def hunt(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for subdomain takeover vulnerabilities using BBOT discovery."""
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided. Use --targets to specify targets.")
        return

    console.print(f"Starting subdomain takeover hunt on {len(targets)} targets")

    tool = await BBotTool.create()
    events = tool.run(
        targets=targets,
        presets=presets,
        modules=modules,
        flags=flags,
        config=config,
    )

    async for event in events:
        console.print(event)
        
        # Analyze DNS_NAME events for takeover vulnerabilities
        if event.type == "DNS_NAME":
            try:
                subdomain = str(event.data)
                console.print(f"Analyzing subdomain: {subdomain}")
                
                takeover_agent = create_takeover_agent()
                
                result = await takeover_agent.run(
                    f"Analyze the subdomain '{subdomain}' for potential takeover vulnerabilities. "
                    f"Use your tools as needed and provide a concise risk assessment."
                )
                
                display_analysis_result(result, subdomain)
                
            except Exception as e:
                console.print(f"Error analyzing subdomain: {e}")

        # Analyze FINDING events for takeover indicators
        elif event.type == "FINDING" and any(keyword in str(event.data).lower() for keyword in ["takeover", "dangling", "cname"]):
            try:
                console.print(f"BBOT finding: {event.data}")
                
                # Extract domains from finding
                domain_pattern = r'([a-zA-Z0-9]([a-zA-Z0-9-]{1,61})?[a-zA-Z0-9]\.)+[a-zA-Z]{2,}'
                domains = re.findall(domain_pattern, str(event.data))
                
                if domains:
                    domain = domains[0][0] if isinstance(domains[0], tuple) else domains[0]
                    console.print(f"Validating extracted domain: {domain}")
                    
                    takeover_agent = create_takeover_agent()
                    result = await takeover_agent.run(
                        f"Validate the potential subdomain takeover for '{domain}'. "
                        f"Determine if this is a genuine vulnerability."
                    )
                    
                    display_analysis_result(result, domain)
                else:
                    console.print("No domain extracted from finding")
                    
            except Exception as e:
                console.print(f"Error validating finding: {e}")


@app.command 
async def validate(subdomain: str) -> None:
    """Validate a specific subdomain for takeover vulnerability."""
    
    console.print(f"Validating subdomain: {subdomain}")
    
    try:
        takeover_agent = create_takeover_agent()
        
        result = await takeover_agent.run(
            f"Analyze the subdomain '{subdomain}' for potential takeover vulnerabilities. "
            f"Use your tools strategically to assess risks and provide actionable recommendations."
        )
        
        display_analysis_result(result, subdomain)
        
    except Exception as e:
        console.print(f"Validation failed: {e}")


if __name__ == "__main__":
    app()