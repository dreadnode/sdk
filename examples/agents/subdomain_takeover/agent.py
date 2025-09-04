import argparse
import asyncio
import json
import re
import time
import typing as t
from pathlib import Path

from rich.console import Console

import dreadnode as dn
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
        import pydantic.dataclasses
        pydantic.dataclasses.rebuild_dataclass(event_class)
except Exception:
    pass

# Configure Dreadnode
dn.configure(server=None, token=None, project="subdomain-takeover-agent", console=False)

console = Console()


@dn.task(name="Analyze Subdomain", label="analyze_subdomain")
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
            tool_name = getattr(message, "name", "unknown")
            tool_outputs[tool_name] = message.content
            dn.log_output(f"tool_output_{tool_name}", message.content)

            if "Commands executed:" in message.content:
                commands_section = message.content.split("Commands executed:")[1].split("Results:")[
                    0
                ]
                commands = [
                    line.strip() for line in commands_section.strip().split("\n") if line.strip()
                ]
                dn.log_output(f"executed_commands_{tool_name}", commands)

    finding_stored = "store_subdomain_takeover_finding" in tools_used
    has_finding = finding_stored
    if result.messages and result.messages[-1].content:
        has_finding = has_finding or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "potential takeover",
                "subdomain takeover vulnerability",
                "takeover vulnerability",
                "vulnerable to takeover",
                "dangling cname",
                "unclaimed resource",
                "takeover indicator",
                "successful subdomain takeover",
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_finding", 1 if has_finding else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)

    analysis_result = {
        "subdomain": subdomain,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_finding": has_finding,
        "stored_in_db": finding_stored,
    }

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


def is_subdomain_takeover_event(event: dict) -> bool:
    """Check if a BBOT event is related to potential subdomain takeover."""
    event_type = event.get('type')
    
    # Handle DNS_NAME events with CNAME records
    if event_type == 'DNS_NAME':
        dns_children = event.get('dns_children', {})
        cname_records = dns_children.get('CNAME', [])
        
        if not cname_records:
            return False
        
        # Check for cloud service indicators in CNAME targets
        cloud_indicators = [
            'amazonaws.com', 'elb.amazonaws.com', 's3.amazonaws.com',
            'azurewebsites.net', 'cloudfront.net', 'herokuapp.com',
            'github.io', 'netlify.com', 'vercel.app', 'surge.sh',
            'shopify.com', 'myshopify.com', 'fastly.com'
        ]
        
        for cname in cname_records:
            if any(indicator in cname.lower() for indicator in cloud_indicators):
                return True
    
    # Handle VULNERABILITY events from baddns module
    elif event_type == 'VULNERABILITY':
        description = event.get('data', {}).get('description', '').lower()
        tags = event.get('tags', [])
        
        # Look for NS record issues that could lead to subdomain takeover
        ns_indicators = [
            'dangling ns', 'ns records without soa', 'baddns-ns'
        ]
        
        if any(indicator in description for indicator in ns_indicators) or 'baddns-ns' in tags:
            return True
    
    return False


@dn.task(name="Analyze Event for Subdomain Takeover", label="analyze_event_takeover")
async def analyze_dns_event(event_data: dict) -> dict:
    """Analyze a BBOT event for subdomain takeover vulnerability."""
    takeover_agent = create_takeover_agent()
    
    event_type = event_data.get('type')
    host = event_data.get('host', '')
    
    if event_type == 'DNS_NAME':
        subdomain = event_data.get('data', '')
        dns_children = event_data.get('dns_children', {})
        cname_records = dns_children.get('CNAME', [])
        
        console.print(f"[*] Analyzing DNS_NAME event for subdomain takeover on {host}")
        console.print(f"    Subdomain: {subdomain}")
        console.print(f"    CNAME targets: {', '.join(cname_records)}")
        
        prompt = (
            f"Analyze the subdomain '{subdomain}' for subdomain takeover vulnerability. "
            f"The subdomain has CNAME records pointing to: {', '.join(cname_records)}. "
            f"Use the tools available to you to test if this subdomain can be taken over."
        )
        
    elif event_type == 'VULNERABILITY':
        subdomain = host
        description = event_data.get('data', {}).get('description', '')
        severity = event_data.get('data', {}).get('severity', '')
        
        console.print(f"[*] Analyzing VULNERABILITY event for subdomain takeover on {host}")
        console.print(f"    Subdomain: {subdomain}")
        console.print(f"    Severity: {severity}")
        console.print(f"    Description: {description[:100]}...")
        
        # Extract NS records from description if available
        ns_records = []
        if 'trigger:' in description.lower():
            trigger_part = description.split('Trigger: [')[1].split(']')[0] if 'Trigger: [' in description else ''
            if trigger_part:
                ns_records = [ns.strip() for ns in trigger_part.split(',')]
        
        prompt = (
            f"Analyze the subdomain '{subdomain}' for NS record takeover vulnerability. "
            f"BBOT detected: {description}. "
            f"This indicates dangling NS records without SOA records. "
            f"NS records found: {', '.join(ns_records)}. "
            f"Use DNS tools to verify if NS records exist but no SOA record is present, "
            f"which could indicate a zone takeover opportunity."
        )
    else:
        raise ValueError(f"Unsupported event type: {event_type}")
    
    result = await takeover_agent.run(prompt)

    tool_outputs = {}
    tools_used = []
    
    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tools_used.append(tool_name)
                console.print(f"[*] Agent calling tool: {tool_name}")
                console.print(f"    Arguments: {tool_call.function.arguments}")
        elif message.role == "tool":
            tool_name = getattr(message, "name", "unknown")
            tool_outputs[tool_name] = message.content
            console.print(f"[*] Tool {tool_name} output:")
            console.print(f"    {message.content[:200]}...")
            dn.log_output(f"tool_output_{tool_name}", message.content)

    finding_stored = "store_subdomain_takeover_finding" in tools_used
    has_takeover = finding_stored
    if result.messages and result.messages[-1].content:
        has_takeover = has_takeover or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "subdomain takeover confirmed",
                "takeover vulnerability confirmed", 
                "vulnerable to takeover",
                "dangling cname",
                "unclaimed resource",
                "takeover possible",
                "subdomain can be taken over"
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_takeover", 1 if has_takeover else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)
    
    # Build analysis result based on event type
    analysis_result = {
        "event_type": event_type,
        "subdomain": subdomain,
        "host": host,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_takeover": has_takeover,
        "stored_in_db": finding_stored,
        "original_event": event_data
    }
    
    # Add event-specific fields
    if event_type == 'DNS_NAME':
        analysis_result["cname_targets"] = cname_records
    elif event_type == 'VULNERABILITY':
        analysis_result["vulnerability_description"] = description
        analysis_result["severity"] = severity
        analysis_result["ns_records"] = ns_records

    return analysis_result


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
            tool_name = getattr(message, "name", "unknown")
            tool_outputs[tool_name] = message.content

    if tools_used:
        console.print(f"Agent used: {', '.join(tools_used)}")

    # Show raw tool outputs in debug mode
    if debug and tool_outputs:
        console.print("\n[DEBUG] Raw tool outputs:")
        for tool_name, output in tool_outputs.items():
            console.print(
                f"  {tool_name}: {output[:200]}..."
                if len(output) > 200
                else f"  {tool_name}: {output}"
            )

    final_message = result.messages[-1]
    if final_message.content:
        console.print(f"\nAnalysis for {subdomain}:")
        console.print(final_message.content)
        console.print(f"\nProcessed {len(result.messages)} messages in {result.steps} steps")


def display_analysis_result_from_task(analysis_result: dict, debug: bool = False) -> None:
    """Display analysis result from task."""
    console.print(f"Agent used: {', '.join(analysis_result['tools_used'])}")

    if debug and analysis_result["tool_outputs"]:
        console.print("\n[DEBUG] Raw tool outputs:")
        for tool_name, output in analysis_result["tool_outputs"].items():
            console.print(
                f"  {tool_name}: {output[:200]}..."
                if len(output) > 200
                else f"  {tool_name}: {output}"
            )

    console.print(f"\nAnalysis for {analysis_result['subdomain']}:")
    console.print(analysis_result["analysis"])
    console.print(
        f"\nProcessed {len(analysis_result['tool_outputs'])} tool calls in {analysis_result['steps']} steps"
    )


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
            flags=flags or [],
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

        # Track metrics at task level
        analyzed_count = 0
        findings_count = 0
        findings = []

        # Analyze each subdomain directly (since we already have a list of subdomains)
        for subdomain in targets:
            try:
                console.print(f"Analyzing subdomain: {subdomain}")

                analysis_result = await analyze_subdomain(subdomain)

                console.print(f"Agent used: {', '.join(analysis_result['tools_used'])}")
                console.print(f"\nAnalysis for {subdomain}:")
                console.print(analysis_result["analysis"])
                console.print(
                    f"\nProcessed {len(analysis_result['tool_outputs'])} tool calls in {analysis_result['steps']} steps"
                )

                analyzed_count += 1
                dn.log_metric("subdomains_analyzed", analyzed_count)

                finding_stored = (
                    "store_subdomain_takeover_finding" in analysis_result["tools_used"]
                )

                if finding_stored or (
                    analysis_result["analysis"]
                    and any(
                        phrase in analysis_result["analysis"].lower()
                        for phrase in [
                            "potential takeover",
                            "subdomain takeover vulnerability",
                            "takeover vulnerability",
                            "vulnerable to takeover",
                            "dangling cname",
                            "unclaimed resource",
                            "takeover indicator",
                            "successful subdomain takeover",
                        ]
                    )
                ):
                    findings_count += 1
                    dn.log_metric("findings_found", findings_count)

                    security_finding = {
                        "subdomain": subdomain,
                        "finding_type": "subdomain_takeover",
                        "risk_level": "high",
                        "analysis": analysis_result["analysis"],
                        "tool_outputs": analysis_result["tool_outputs"],
                        "steps": analysis_result["steps"],
                        "timestamp": time.time(),
                        "stored_in_db": finding_stored,
                    }
                    findings.append(security_finding)
                    dn.log_output(f"finding_{subdomain}", security_finding)

            except Exception as e:
                console.print(f"Error analyzing subdomain: {e}")

        dn.log_metric("subdomains_analyzed", analyzed_count)
        dn.log_metric("findings_found", findings_count)
        dn.log_output("security_findings", findings)
        dn.log_output(
            "summary",
            {
                "total_targets": len(targets),
                "subdomains_analyzed": analyzed_count,
                "findings_found": findings_count,
                "findings": findings,
            },
        )

        console.print("\n📊 Task Summary:")
        console.print(f"   Subdomains analyzed: {analyzed_count}")
        console.print(f"   Security findings: {findings_count}")


async def modules() -> None:
    """List available BBOT modules."""
    BBotTool.get_modules()


async def presets() -> None:
    """List available BBOT presets."""
    BBotTool.get_presets()


async def flags() -> None:
    """List available BBOT flags."""
    BBotTool.get_flags()


async def events() -> None:
    """List available BBOT event types."""
    BBotTool.get_events()


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

            finding_stored = "store_subdomain_takeover_finding" in analysis_result["tools_used"]
            has_finding = finding_stored or (
                analysis_result["analysis"]
                and any(
                    phrase in analysis_result["analysis"].lower()
                    for phrase in [
                        "potential takeover",
                        "subdomain takeover vulnerability",
                        "takeover vulnerability",
                        "vulnerable to takeover",
                        "dangling cname",
                        "unclaimed resource",
                        "takeover indicator",
                        "successful subdomain takeover",
                    ]
                )
            )

            if has_finding:
                security_finding = {
                    "subdomain": subdomain,
                    "finding_type": "subdomain_takeover",
                    "risk_level": "high",
                    "analysis": analysis_result["analysis"],
                    "tool_outputs": analysis_result["tool_outputs"],
                    "steps": analysis_result["steps"],
                    "timestamp": time.time(),
                    "stored_in_db": finding_stored,
                }
                dn.log_output("security_finding", security_finding)

            dn.log_output(
                "analysis_result",
                {
                    "subdomain": subdomain,
                    "has_finding": has_finding,
                    "analysis": analysis_result["analysis"],
                    "tool_outputs": analysis_result["tool_outputs"],
                    "steps": analysis_result["steps"],
                },
            )
            dn.log_metric("findings_found", 1 if has_finding else 0)
            dn.log_metric("subdomains_analyzed", 1)

        except Exception as e:
            console.print(f"Validation failed: {e}")
            dn.log_output("error", str(e))


async def analyze_dns_events_file(dns_events_file: Path, debug: bool = False) -> None:
    """Analyze DNS_NAME events from a JSON file for subdomain takeover vulnerabilities."""
    
    with dn.run("dns-events-analysis"):
        console.print(f"Analyzing DNS events from {dns_events_file}")
        
        try:
            with open(dns_events_file) as f:
                events = json.load(f)
            
            if not isinstance(events, list):
                events = [events]
            
            takeover_count = 0
            total_dns_events = 0
            
            for event in events:
                if is_subdomain_takeover_event(event):
                    total_dns_events += 1
                    console.print(f"[*] Found potential subdomain takeover DNS event...")
                    
                    analysis_result = await analyze_dns_event(event)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                        if analysis_result['event_type'] == 'DNS_NAME':
                            console.print(f"CNAME targets: {', '.join(analysis_result['cname_targets'])}")
                        elif analysis_result['event_type'] == 'VULNERABILITY':
                            console.print(f"Vulnerability: {analysis_result['vulnerability_description'][:100]}...")
                        console.print(f"Analysis: {analysis_result['analysis'][:200]}...")
                    
                    if analysis_result["has_takeover"]:
                        takeover_count += 1
                        console.print(f"SUBDOMAIN TAKEOVER CONFIRMED: {analysis_result['subdomain']}")
                    else:
                        console.print(f"No subdomain takeover found for {analysis_result['subdomain']}")
            
            dn.log_metric("dns_events_analyzed", total_dns_events)
            dn.log_metric("takeover_vulnerabilities", takeover_count)
            
            console.print(f"\nDNS Events Analysis Summary:")
            console.print(f"   DNS events with takeover indicators: {total_dns_events}")
            console.print(f"   Confirmed subdomain takeover vulnerabilities: {takeover_count}")
            
        except Exception as e:
            console.print(f"Error analyzing DNS events file: {e}")


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

    # Analyze DNS events command
    analyze_parser = subparsers.add_parser("analyze-dns", help="Analyze DNS_NAME events from BBOT for subdomain takeover")
    analyze_parser.add_argument("dns_events_file", type=Path, help="JSON file containing BBOT DNS_NAME events")
    analyze_parser.add_argument("--debug", action="store_true", help="Show debug information")

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
    elif args.command == "analyze-dns":
        await analyze_dns_events_file(args.dns_events_file, args.debug)
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
