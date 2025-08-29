import argparse
import asyncio
import re
import time
import typing as t
from pathlib import Path

import dreadnode as dn
import pydantic.dataclasses
from rich.console import Console

from dreadnode.agent.agent import Agent
from dreadnode.agent.result import AgentResult
from dreadnode.agent.tools.bbot.tool import BBotTool
from dreadnode.agent.tools.kali.tool import KaliTool
from dreadnode.agent.tools.neo4j.tool import Neo4jTool

# Import necessary components for Pydantic dataclass fix
from dreadnode.agent.events import (
    AgentEnd,
    AgentError,
    AgentStalled,
    AgentStart,
    Event,
    GenerationEnd,
    StepStart,
    ToolEnd,
    ToolStart,
)

# Rebuild dataclasses after all imports are complete
try:
    from dreadnode.agent.state import State
    from dreadnode.agent.reactions import Reaction
    
    critical_classes = [
        Event,
        AgentStart,
        StepStart,
        GenerationEnd,
        AgentStalled,
        AgentError,
        ToolStart,
        ToolEnd,
        AgentEnd,
    ]

    for event_class in critical_classes:
        pydantic.dataclasses.rebuild_dataclass(event_class)
except Exception:
    pass


"""Usage:
uv run python examples/agents/subdomain_takeover/agent.py validate test.example.com
uv run python examples/agents/subdomain_takeover/agent.py hunt --targets "/path/to/dir/subdomains.txt"
"""


# Configure Dreadnode
dn.configure(server=None, token=None, project="subdomain-takeover-agent", console=False)


console = Console()


@dn.task()
async def analyze_subdomain(subdomain: str) -> dict:
    """Analyze a single subdomain for takeover vulnerabilities."""
    takeover_agent = create_takeover_agent()
    
    result = await takeover_agent.run(
        f"Analyze the subdomain '{subdomain}' for potential takeover vulnerabilities. "
        f"Use your tools as needed and provide a concise risk assessment."
    )
    
    tool_outputs = {}
    tools_used = []
    
    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tools_used.append(tool_call.function.name)
        elif message.role == "tool":
            tool_name = getattr(message, 'name', 'unknown')
            tool_outputs[tool_name] = message.content
            dn.log_output(f"tool_output_{tool_name}", message.content)
    
    analysis_result = {
        "subdomain": subdomain,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps
    }
    
    dn.log_metric("tools_used_count", len(tools_used))
    dn.log_output("raw_tool_data", tool_outputs)
    
    return analysis_result


def create_takeover_agent() -> Agent:
    """Create a subdomain takeover analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool()]

    return Agent(
        name="subdomain-takeover-agent",
        description="An agent that detects and stores subdomain takeover vulnerabilities",
        model="gpt-4",
        tools=tools,
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

IMPORTANT: If CNAME resolves to A record owned by target organization, takeover is highly unlikely.

Example vulnerability:
marketing.example.com → CNAME → myapp.herokudns.com (but myapp is deleted/unclaimed)

Report ONLY actual takeover vulnerabilities, not general DNS misconfigurations.

When you find CONFIRMED takeover vulnerabilities, store them using Neo4jTool.store_subdomain_takeover_finding(subdomain, vulnerability_type, risk_level, cname_target, error_message, service_provider).""",
    )


def display_analysis_result(result: AgentResult, subdomain: str, debug: bool = False) -> None:
    """Display the agent's analysis result."""
    if not result or not result.messages:
        console.print("Analysis completed but no content available")
        return

    # Show which tools the agent decided to use
    tools_used = []
    tool_outputs = {}
    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tools_used.append(tool_call.function.name)
        elif message.role == "tool" and debug:
            # Capture tool outputs for debugging
            tool_name = getattr(message, 'name', 'unknown')
            tool_outputs[tool_name] = message.content

    if tools_used:
        console.print(f"Agent used: {', '.join(tools_used)}")

    # Show raw tool outputs in debug mode
    if debug and tool_outputs:
        console.print(f"\n[DEBUG] Raw tool outputs:")
        for tool_name, output in tool_outputs.items():
            console.print(f"  {tool_name}: {output[:200]}..." if len(output) > 200 else f"  {tool_name}: {output}")

    final_message = result.messages[-1]
    if final_message.content:
        console.print(f"\nAnalysis for {subdomain}:")
        console.print(final_message.content)
        console.print(f"\nProcessed {len(result.messages)} messages in {result.steps} steps")


def display_analysis_result_from_task(analysis_result: dict, debug: bool = False) -> None:
    """Display analysis result from task."""
    console.print(f"Agent used: {', '.join(analysis_result['tools_used'])}")

    if debug and analysis_result['tool_outputs']:
        console.print(f"\n[DEBUG] Raw tool outputs:")
        for tool_name, output in analysis_result['tool_outputs'].items():
            console.print(f"  {tool_name}: {output[:200]}..." if len(output) > 200 else f"  {tool_name}: {output}")

    console.print(f"\nAnalysis for {analysis_result['subdomain']}:")
    console.print(analysis_result['analysis'])
    console.print(f"\nProcessed {len(analysis_result['tool_outputs'])} tool calls in {analysis_result['steps']} steps")


async def modules() -> None:
    """List available BBOT modules."""
    tool = await BBotTool.create()
    tool.get_modules()


async def presets() -> None:
    """List available BBOT presets."""
    tool = await BBotTool.create()
    tool.get_presets()


async def flags() -> None:
    """List available BBOT flags."""
    tool = await BBotTool.create()
    tool.get_flags()


async def events() -> None:
    """List available BBOT event types."""
    tool = await BBotTool.create()
    tool.get_events()


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

    # Start dreadnode run context
    with dn.run("subdomain-takeover-hunt"):
        # Log parameters
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or []
        )
        
        # Log inputs
        dn.log_input("targets", targets)
        if presets:
            dn.log_input("presets", presets)
        if modules:
            dn.log_input("modules", modules)
        if flags:
            dn.log_input("flags", flags)
        if config:
            dn.log_input("config", str(config))
        
        console.print(f"Starting subdomain takeover hunt on {len(targets)} targets")

        tool = await BBotTool.create()
        events = tool.run(
            targets=targets,
            presets=presets,
            modules=modules,
            flags=flags,
            config=config,
        )

        # Track metrics at task level
        analyzed_count = 0
        findings_count = 0
        findings = []

        async for event in events:
            console.print(event)

            # Analyze DNS_NAME events for takeover vulnerabilities
            if event.type == "DNS_NAME":
                try:
                    subdomain = str(event.data)
                    console.print(f"Analyzing subdomain: {subdomain}")

                    analysis_result = await analyze_subdomain(subdomain)
                    
                    console.print(f"Agent used: {', '.join(analysis_result['tools_used'])}")
                    console.print(f"\nAnalysis for {subdomain}:")
                    console.print(analysis_result['analysis'])
                    console.print(f"\nProcessed {len(analysis_result['tool_outputs'])} tool calls in {analysis_result['steps']} steps")
                    
                    analyzed_count += 1
                    dn.log_metric("subdomains_analyzed", analyzed_count)
                    
                    finding_stored = "store_subdomain_takeover_finding" in analysis_result['tools_used']
                    
                    if finding_stored or (analysis_result['analysis'] and any(
                        phrase in analysis_result['analysis'].lower() 
                        for phrase in [
                            "potential takeover", "subdomain takeover vulnerability", "takeover vulnerability",
                            "vulnerable to takeover", "dangling cname", "unclaimed resource", 
                            "takeover indicator", "successful subdomain takeover"
                        ]
                    )):
                        findings_count += 1
                        dn.log_metric("findings_found", findings_count)
                        
                        security_finding = {
                            "subdomain": subdomain,
                            "finding_type": "subdomain_takeover",
                            "risk_level": "high",
                            "analysis": analysis_result['analysis'],
                            "tool_outputs": analysis_result['tool_outputs'],
                            "steps": analysis_result['steps'],
                            "timestamp": time.time(),
                            "stored_in_db": finding_stored
                        }
                        findings.append(security_finding)
                        dn.log_output(f"finding_{subdomain}", security_finding)

                except Exception as e:
                    console.print(f"Error analyzing subdomain: {e}")

            # Analyze FINDING events for takeover indicators
            elif event.type == "FINDING" and any(
                keyword in str(event.data).lower() for keyword in ["takeover", "dangling", "cname"]
            ):
                try:
                    console.print(f"BBOT finding: {event.data}")

                    # Extract domains from finding
                    domain_pattern = r"([a-zA-Z0-9]([a-zA-Z0-9-]{1,61})?[a-zA-Z0-9]\.)+[a-zA-Z]{2,}"
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
                        analyzed_count += 1
                    else:
                        console.print("No domain extracted from finding")

                except Exception as e:
                    console.print(f"Error validating finding: {e}")
        
        dn.log_metric("subdomains_analyzed", analyzed_count)
        dn.log_metric("findings_found", findings_count)
        dn.log_output("security_findings", findings)
        dn.log_output("summary", {
            "total_targets": len(targets),
            "subdomains_analyzed": analyzed_count,
            "findings_found": findings_count,
            "findings": findings
        })
        
        console.print(f"\n📊 Task Summary:")
        console.print(f"   Subdomains analyzed: {analyzed_count}")
        console.print(f"   Security findings: {findings_count}")


async def validate(subdomain: str, debug: bool = False) -> None:
    """Validate a specific subdomain for takeover vulnerability."""

    # Start dreadnode run context
    with dn.run("subdomain-takeover-validate"):
        # Log parameters
        dn.log_params(subdomain=subdomain)
        
        # Log inputs
        dn.log_input("subdomain", subdomain)
        
        console.print(f"Validating subdomain: {subdomain}")

        try:
            analysis_result = await analyze_subdomain(subdomain)
            
            display_analysis_result_from_task(analysis_result, debug=debug)
            
            finding_stored = "store_subdomain_takeover_finding" in analysis_result['tools_used']
            has_finding = finding_stored or (analysis_result['analysis'] and any(
                phrase in analysis_result['analysis'].lower() 
                for phrase in [
                    "potential takeover", "subdomain takeover vulnerability", "takeover vulnerability",
                    "vulnerable to takeover", "dangling cname", "unclaimed resource", 
                    "takeover indicator", "successful subdomain takeover"
                ]
            ))
            
            if has_finding:
                security_finding = {
                    "subdomain": subdomain,
                    "finding_type": "subdomain_takeover", 
                    "risk_level": "high",
                    "analysis": analysis_result['analysis'],
                    "tool_outputs": analysis_result['tool_outputs'],
                    "steps": analysis_result['steps'],
                    "timestamp": time.time(),
                    "stored_in_db": finding_stored
                }
                dn.log_output("security_finding", security_finding)
 
            dn.log_output("analysis_result", {
                "subdomain": subdomain,
                "has_finding": has_finding,
                "analysis": analysis_result['analysis'],
                "tool_outputs": analysis_result['tool_outputs'],
                "steps": analysis_result['steps']
            })
            dn.log_metric("findings_found", 1 if has_finding else 0)
            dn.log_metric("subdomains_analyzed", 1)

        except Exception as e:
            console.print(f"Validation failed: {e}")
            dn.log_output("error", str(e))


async def main():
    parser = argparse.ArgumentParser(description="Subdomain takeover vulnerability scanner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Hunt command
    hunt_parser = subparsers.add_parser("hunt", help="Hunt for subdomain takeover vulnerabilities")
    hunt_parser.add_argument(
        "--targets", type=Path, help="Path to file containing target subdomains"
    )
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a specific subdomain")
    validate_parser.add_argument("subdomain", help="Subdomain to validate")
    validate_parser.add_argument("--debug", action="store_true", help="Show raw tool outputs")

    # Info commands
    subparsers.add_parser("modules", help="List available BBOT modules")
    subparsers.add_parser("presets", help="List available BBOT presets")
    subparsers.add_parser("flags", help="List available BBOT flags")
    subparsers.add_parser("events", help="List available BBOT event types")

    args = parser.parse_args()

    if args.command == "hunt":
        await hunt(args.targets, args.presets, args.modules, args.flags, args.config)
    elif args.command == "validate":
        await validate(args.subdomain, args.debug)
    elif args.command == "modules":
        await modules()
    elif args.command == "presets":
        await presets()
    elif args.command == "flags":
        await flags()
    elif args.command == "events":
        await events()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
