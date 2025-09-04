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

dn.configure(server=None, token=None, project="host-header-hunter-agent", console=False)

console = Console()


@dn.task(name="Analyze Host Header Finding", label="analyze_host_header_finding")
async def analyze_host_header_finding(finding_data: dict) -> dict:
    """Analyze a BBOT host header injection finding for exploitability."""
    host_header_agent = create_host_header_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    console.print(f"[*] Analyzing host header injection finding on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Description: {description}")
    
    result = await host_header_agent.run(
        f"Analyze the potential host header injection vulnerability at {url}. "
        f"Test for host header injection by modifying the Host header with various payloads. "
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

    finding_stored = "store_host_header_finding" in tools_used
    has_host_header_injection = finding_stored
    if result.messages and result.messages[-1].content:
        has_host_header_injection = has_host_header_injection or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "host header injection confirmed",
                "host header reflected",
                "injection successful",
                "host header vulnerability confirmed",
                "malicious host reflected",
                "host header injection detected",
                "vulnerable to host header injection",
                "host header poisoning",
                "cache poisoning possible"
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_host_header_injection", 1 if has_host_header_injection else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)
    
    analysis_result = {
        "url": url,
        "host": host,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_host_header_injection": has_host_header_injection,
        "stored_in_db": finding_stored,
        "original_finding": finding_data
    }

    return analysis_result


def create_host_header_agent() -> Agent:
    """Create a host header injection vulnerability analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool(), OastTool()]

    return Agent(
        name="host-header-hunter-agent",
        description="An agent that analyzes and exploits host header injection vulnerabilities",
        model="gpt-4-turbo",
        tools=tools,
        instructions="""You are an expert at analyzing and exploiting host header injection vulnerabilities.

Use the tools available to you to test for host header injection by modifying the Host header with various payloads.

Analyze responses for indicators of host header injection such as:
- Host header values reflected in response body or headers
- Different response codes or content when host header is modified
- Cache poisoning indicators
- Password reset poisoning possibilities

Be thorough and systematic in your testing approach. Use your expertise to choose appropriate payloads and techniques.

If you confirm host header injection exists, use store_host_header_finding to record the vulnerability with details about the successful payload and evidence.""",
    )


def is_host_header_finding(event: dict) -> bool:
    """Check if a BBOT event is a host header injection finding."""
    if event.get('type') != 'FINDING':
        return False
    
    description = event.get('data', {}).get('description', '')
    return 'Host Header Injection' in description or 'host header' in description.lower()


async def hunt_from_bbot_scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for host header injection vulnerabilities from BBOT scan findings."""
    
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided. Use --targets to specify targets.")
        return

    with dn.run("host-header-hunt-from-bbot"):
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
        )

        console.print(f"Starting host header injection hunt on {len(targets)} targets using BBOT scan...")

        host_header_findings_count = 0
        total_findings = 0
        
        tool = BBotTool()
        scan_modules = modules or ["httpx", "hunt"]
        
        for target in targets:
            try:
                console.print(f"[*] Scanning {target} for host header injection...")
                
                scan_config = config or {"omit_event_types": []}
                
                events = tool.run(
                    target=target,
                    presets=presets,
                    modules=scan_modules,
                    flags=flags,
                    config=scan_config,
                )
                
                async for event in events:
                    if is_host_header_finding(event):
                        total_findings += 1
                        console.print(f"Found host header injection candidate on {event.get('host')}")
                        
                        try:
                            analysis_result = await analyze_host_header_finding(event)
                            
                            if analysis_result["has_host_header_injection"]:
                                host_header_findings_count += 1
                                
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"], 
                                    "finding_type": "host_header_injection",
                                    "risk_level": "medium",
                                    "analysis": analysis_result["analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": time.time(),
                                    "stored_in_db": analysis_result["stored_in_db"],
                                }
                                
                                dn.log_output(f"host_header_finding_{analysis_result['host']}", security_finding)
                                console.print(f"HOST HEADER INJECTION CONFIRMED on {analysis_result['host']}")
                            else:
                                console.print(f"Host header injection not exploitable on {event.get('host')}")
                                
                        except Exception as e:
                            console.print(f"Error analyzing host header injection finding: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")

        dn.log_metric("total_findings", total_findings)
        dn.log_metric("host_header_confirmed", host_header_findings_count)
        
        console.print(f"\nHunt Summary:")
        console.print(f"   Host header injection candidates found: {total_findings}")
        console.print(f"   Host header injection vulnerabilities confirmed: {host_header_findings_count}")


async def analyze_finding_file(finding_file: Path, debug: bool = False) -> None:
    """Analyze host header injection findings from a JSON file (for testing)."""
    
    with dn.run("host-header-analyze-findings"):
        console.print(f"Analyzing findings from {finding_file}")
        
        try:
            with open(finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            host_header_count = 0
            for finding in findings:
                if is_host_header_finding(finding):
                    console.print(f"[*] Analyzing host header injection finding...")
                    analysis_result = await analyze_host_header_finding(finding)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                        console.print(f"Analysis: {analysis_result['analysis'][:200]}...")
                    
                    if analysis_result["has_host_header_injection"]:
                        host_header_count += 1
                        console.print(f"HOST HEADER INJECTION CONFIRMED!")
                    else:
                        console.print(f"No host header injection exploitation possible")
            
            dn.log_metric("host_header_findings", host_header_count)
            console.print(f"\nAnalysis Summary:")
            console.print(f"   Host header injection vulnerabilities confirmed: {host_header_count}")
            
        except Exception as e:
            console.print(f"Error analyzing findings file: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Host header injection vulnerability hunter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hunt_parser = subparsers.add_parser("hunt", help="Hunt for host header injection vulnerabilities using BBOT")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use (default: httpx,hunt)")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze host header injection findings from JSON file")
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