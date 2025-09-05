import argparse
import asyncio
import json
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
from dreadnode.agent.tools.oast.tool import OastTool

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

dn.configure(server=None, token=None, project="graphql-hunter-agent", console=False)

console = Console()


@dn.task(name="Analyze GraphQL Finding", label="analyze_graphql_finding")
async def analyze_graphql_finding(finding_data: dict) -> dict:
    """Analyze a BBOT GraphQL introspection finding for exploitability."""
    graphql_agent = create_graphql_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    console.print(f"[*] Analyzing GraphQL introspection finding on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Description: {description}")
    
    result = await graphql_agent.run(
        f"Analyze the GraphQL introspection vulnerability at {url}. "
        f"Test the GraphQL endpoint for introspection capabilities and enumerate the schema. "
        f"Use the tools available to you to test systematically using your expertise."
    )

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

    finding_stored = "store_graphql_finding" in tools_used
    endpoint_stored = "store_graphql_endpoint" in tools_used
    has_graphql_introspection = finding_stored or endpoint_stored
    if result.messages and result.messages[-1].content:
        has_graphql_introspection = has_graphql_introspection or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "introspection enabled",
                "graphql schema exposed",
                "introspection query successful",
                "schema introspection available",
                "graphql introspection confirmed",
                "schema enumeration successful",
                "types and fields exposed",
                "graphql vulnerability confirmed",
                "introspection not disabled"
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_graphql_introspection", 1 if has_graphql_introspection else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_metric("endpoint_stored", 1 if endpoint_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)
    
    analysis_result = {
        "url": url,
        "host": host,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_graphql_introspection": has_graphql_introspection,
        "stored_in_db": finding_stored,
        "endpoint_stored": endpoint_stored,
        "original_finding": finding_data
    }

    return analysis_result


def create_graphql_agent() -> Agent:
    """Create a GraphQL introspection vulnerability analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool(), OastTool()]

    return Agent(
        name="graphql-hunter-agent",
        description="An agent that analyzes and exploits GraphQL introspection vulnerabilities",
        model="gpt-4-turbo",
        tools=tools,
        instructions="""You are an expert at analyzing and exploiting GraphQL introspection vulnerabilities.

Use the tools available to you to test for GraphQL introspection by sending introspection queries to the GraphQL endpoint.

Analyze responses for indicators of GraphQL introspection such as:
- Successful introspection queries revealing schema information
- Type definitions and field information exposed
- Query and mutation structures revealed
- Sensitive field names or types that could indicate data exposure risks

Be thorough and systematic in your testing approach. Use your expertise to craft appropriate introspection queries and analyze the responses.

If you successfully retrieve GraphQL schema information:
1. Use store_graphql_endpoint to record the endpoint with detailed schema data including types, queries, mutations and their relationships
2. If this represents a security vulnerability (introspection enabled in production), also use store_graphql_finding to record the vulnerability finding

Focus on extracting and storing comprehensive schema information for analysis.""",
    )


def is_graphql_finding(event: dict) -> bool:
    """Check if a BBOT event is a GraphQL introspection finding."""
    if event.get('type') != 'FINDING':
        return False
    
    description = event.get('data', {}).get('description', '')
    return 'GraphQL' in description or 'introspection' in description.lower()


async def hunt_from_bbot_scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for GraphQL introspection vulnerabilities from BBOT scan findings."""
    
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided. Use --targets to specify targets.")
        return

    with dn.run("graphql-hunt-from-bbot"):
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
        )

        console.print(f"Starting GraphQL introspection hunt on {len(targets)} targets using BBOT scan...")

        graphql_findings_count = 0
        total_findings = 0
        
        tool = BBotTool()
        scan_modules = modules or ["httpx", "graphql_introspection"]
        
        for target in targets:
            try:
                console.print(f"[*] Scanning {target} for GraphQL introspection...")
                
                scan_config = config or {"omit_event_types": []}
                
                events = tool.run(
                    target=target,
                    presets=presets,
                    modules=scan_modules,
                    flags=flags,
                    config=scan_config,
                )
                
                async for event in events:
                    if is_graphql_finding(event):
                        total_findings += 1
                        console.print(f"Found GraphQL introspection candidate on {event.get('host')}")
                        
                        try:
                            analysis_result = await analyze_graphql_finding(event)
                            
                            if analysis_result["has_graphql_introspection"]:
                                graphql_findings_count += 1
                                
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"], 
                                    "finding_type": "graphql_introspection",
                                    "risk_level": "medium",
                                    "analysis": analysis_result["analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": time.time(),
                                    "stored_in_db": analysis_result["stored_in_db"],
                                }
                                
                                dn.log_output(f"graphql_finding_{analysis_result['host']}", security_finding)
                                console.print(f"GRAPHQL INTROSPECTION CONFIRMED on {analysis_result['host']}")
                            else:
                                console.print(f"GraphQL introspection not exploitable on {event.get('host')}")
                                
                        except Exception as e:
                            console.print(f"Error analyzing GraphQL introspection finding: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")

        dn.log_metric("total_findings", total_findings)
        dn.log_metric("graphql_confirmed", graphql_findings_count)
        
        console.print(f"\nHunt Summary:")
        console.print(f"   GraphQL introspection candidates found: {total_findings}")
        console.print(f"   GraphQL introspection vulnerabilities confirmed: {graphql_findings_count}")


async def analyze_finding_file(finding_file: Path, debug: bool = False) -> None:
    """Analyze GraphQL introspection findings from a JSON file (for testing)."""
    
    with dn.run("graphql-analyze-findings"):
        console.print(f"Analyzing findings from {finding_file}")
        
        try:
            with open(finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            graphql_count = 0
            for finding in findings:
                if is_graphql_finding(finding):
                    console.print(f"[*] Analyzing GraphQL introspection finding...")
                    analysis_result = await analyze_graphql_finding(finding)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                        console.print(f"Analysis: {analysis_result['analysis'][:200]}...")
                    
                    if analysis_result["has_graphql_introspection"]:
                        graphql_count += 1
                        console.print(f"GRAPHQL INTROSPECTION CONFIRMED!")
                    else:
                        console.print(f"No GraphQL introspection exploitation possible")
            
            dn.log_metric("graphql_findings", graphql_count)
            console.print(f"\nAnalysis Summary:")
            console.print(f"   GraphQL introspection vulnerabilities confirmed: {graphql_count}")
            
        except Exception as e:
            console.print(f"Error analyzing findings file: {e}")


async def main():
    parser = argparse.ArgumentParser(description="GraphQL introspection vulnerability hunter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hunt_parser = subparsers.add_parser("hunt", help="Hunt for GraphQL introspection vulnerabilities using BBOT")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use (default: httpx,graphql_introspection)")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze GraphQL introspection findings from JSON file")
    analyze_parser.add_argument("finding_file", type=Path, help="JSON file containing BBOT findings")
    analyze_parser.add_argument("--debug", action="store_true", help="Show debug information")

    args = parser.parse_args()

    if args.command == "hunt":
        await hunt_from_bbot_scan(args.targets, args.presets, args.modules, args.flags, args.config)
    elif args.command == "analyze":
        await analyze_finding_file(args.finding_file, args.debug)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())