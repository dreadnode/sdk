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
dn.configure(server=None, token=None, project="ssrf-hunter-agent", console=False)

console = Console()


@dn.task(name="Analyze SSRF Finding", label="analyze_ssrf_finding")
async def analyze_ssrf_finding(finding_data: dict) -> dict:
    """Analyze a BBOT SSRF finding for exploitability."""
    ssrf_agent = create_ssrf_agent()
    
    # Extract key details from BBOT finding
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    # Parse parameter details from description
    param_name = extract_param_name(description)
    param_type = extract_param_type(description)
    original_value = extract_original_value(description)
    
    console.print(f"[*] Analyzing SSRF finding on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Parameter: {param_name} ({param_type})")
    console.print(f"    Original value preview: {original_value[:50]}...")
    
    result = await ssrf_agent.run(
        f"Analyze the potential SSRF vulnerability at {url} using parameter '{param_name}'. "
        f"The original parameter value was: {original_value[:100]}{'...' if len(original_value) > 100 else ''}\n\n"
        f"Use http_request tool to test SSRF by modifying the {param_name} parameter with different payloads. "
        f"Start with a few basic tests, analyze the responses, then iterate with more targeted payloads based on what you discover."
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

    # Check if SSRF was confirmed based on agent analysis and tool storage
    finding_stored = "store_ssrf_finding" in tools_used
    has_ssrf = finding_stored
    if result.messages and result.messages[-1].content:
        has_ssrf = has_ssrf or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "ssrf confirmed",
                "server-side request forgery",
                "internal service access",
                "cloud metadata accessible", 
                "localhost accessible",
                "internal network response",
                "ssrf vulnerability confirmed",
                "internal service responded",
                "metadata endpoint accessible",
                "ssrf vulnerability detected"
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_ssrf", 1 if has_ssrf else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)
    
    analysis_result = {
        "url": url,
        "host": host,
        "parameter": param_name,
        "param_type": param_type,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_ssrf": has_ssrf,
        "stored_in_db": finding_stored,
        "original_finding": finding_data
    }

    return analysis_result


def create_ssrf_agent() -> Agent:
    """Create an SSRF vulnerability analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool()]

    return Agent(
        name="ssrf-hunter-agent",
        description="An agent that analyzes and exploits SSRF vulnerabilities",
        model="gpt-4-turbo",
        tools=tools,
        instructions="""You are an expert at analyzing and exploiting Server-Side Request Forgery (SSRF) vulnerabilities.

APPROACH: Use http_request tool to test SSRF by crafting URLs with different payloads and analyzing responses for all types of environments, including cloud metadata services for all cloud providers, internal web services, and local resources. Do not just suggest payloads - actually execute requests and analyze results and do not stop until you have been able to execute an exploit, your turns are not limited.

Based on observed results, try additional payloads or techniques for all types of SSRF:
- Blind SSRF: Look for time delays, error messages, or out-of-band interactions.
- Full Response SSRF: Analyze response codes, content, headers, and timing.
- Time-Based SSRF: Use time delays to infer internal service behavior.
- Different response codes (403 vs 200 vs 500) between internal/external
- Response time differences (timeouts on internal vs fast external)
- Error messages revealing internal hostnames/services
- Different content lengths suggesting internal service responses
- Connection errors that indicate network access attempts

When you CONFIRM SSRF, use store_ssrf_finding with:
- The vulnerable URL and parameter
- Vulnerability type: "blind_ssrf", "full_response_ssrf", "time_based_ssrf"  
- Risk level based on what you can access
- The successful payload that proved SSRF
- Evidence from response analysis

IMPORTANT: Use available tools - start with a few payloads, analyze responses, then iterate with more targeted tests based on what you learn.""",
    )


def extract_param_name(description: str) -> str:
    """Extract parameter name from BBOT finding description."""
    if "Name: [" in description:
        start = description.find("Name: [") + 7
        end = description.find("]", start)
        return description[start:end] if end > start else "unknown"
    return "unknown"


def extract_param_type(description: str) -> str:
    """Extract parameter type from BBOT finding description."""
    if "Parameter Type: [" in description:
        start = description.find("Parameter Type: [") + 17
        end = description.find("]", start)
        return description[start:end] if end > start else "unknown"
    return "unknown"


def extract_original_value(description: str) -> str:
    """Extract original parameter value from BBOT finding description."""
    if "Original Value: [" in description:
        start = description.find("Original Value: [") + 17
        end = description.rfind("]")  # Last bracket
        return description[start:end] if end > start else ""
    return ""


def is_ssrf_finding(event: dict) -> bool:
    """Check if a BBOT event is an SSRF finding."""
    if event.get('type') != 'FINDING':
        return False
    
    description = event.get('data', {}).get('description', '')
    return 'Server-side Request Forgery' in description


async def hunt_from_bbot_scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for SSRF vulnerabilities from BBOT scan findings."""
    
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided. Use --targets to specify targets.")
        return

    # Start dreadnode run context
    with dn.run("ssrf-hunt-from-bbot"):
        # Log parameters
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
        )

        console.print(f"Starting SSRF hunt on {len(targets)} targets using BBOT scan...")

        # Track findings
        ssrf_findings_count = 0
        total_findings = 0
        
        # Run BBOT scan with hunt module to find potential SSRF parameters
        tool = BBotTool()
        
        # Use hunt module to find parameters, plus httpx for web crawling
        scan_modules = modules or ["httpx", "excavate", "hunt"]
        
        for target in targets:
            try:
                console.print(f"[*] Scanning {target} for SSRF parameters...")
                
                events = tool.run(
                    target=target,
                    presets=presets,
                    modules=scan_modules,
                    flags=flags,
                    config=config,
                )
                
                async for event in events:
                    # Filter for SSRF findings
                    if is_ssrf_finding(event):
                        total_findings += 1
                        console.print(f"Found SSRF candidate on {event.get('host')}")
                        
                        # Analyze the SSRF finding
                        try:
                            analysis_result = await analyze_ssrf_finding(event)
                            
                            if analysis_result["has_ssrf"]:
                                ssrf_findings_count += 1
                                
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"], 
                                    "parameter": analysis_result["parameter"],
                                    "finding_type": "ssrf",
                                    "risk_level": "high",
                                    "analysis": analysis_result["analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": time.time(),
                                    "stored_in_db": analysis_result["stored_in_db"],
                                }
                                
                                dn.log_output(f"ssrf_finding_{analysis_result['host']}", security_finding)
                                console.print(f"SSRF CONFIRMED on {analysis_result['host']}")
                            else:
                                console.print(f"SSRF not exploitable on {event.get('host')}")
                                
                        except Exception as e:
                            console.print(f"Error analyzing SSRF finding: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")

        dn.log_metric("total_findings", total_findings)
        dn.log_metric("ssrf_confirmed", ssrf_findings_count)
        
        console.print(f"\nHunt Summary:")
        console.print(f"   SSRF candidates found: {total_findings}")
        console.print(f"   SSRF vulnerabilities confirmed: {ssrf_findings_count}")


async def analyze_finding_file(finding_file: Path, debug: bool = False) -> None:
    """Analyze SSRF findings from a JSON file (for testing)."""
    
    with dn.run("ssrf-analyze-findings"):
        console.print(f"Analyzing findings from {finding_file}")
        
        try:
            with open(finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            ssrf_count = 0
            for finding in findings:
                if is_ssrf_finding(finding):
                    console.print(f"[*] Analyzing SSRF finding...")
                    analysis_result = await analyze_ssrf_finding(finding)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                        console.print(f"Analysis: {analysis_result['analysis'][:200]}...")
                    
                    if analysis_result["has_ssrf"]:
                        ssrf_count += 1
                        console.print(f"SSRF CONFIRMED!")
                    else:
                        console.print(f"No SSRF exploitation possible")
            
            dn.log_metric("ssrf_findings", ssrf_count)
            console.print(f"\nAnalysis Summary:")
            console.print(f"   SSRF vulnerabilities confirmed: {ssrf_count}")
            
        except Exception as e:
            console.print(f"Error analyzing findings file: {e}")


async def main():
    parser = argparse.ArgumentParser(description="SSRF vulnerability hunter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Hunt command - scan targets and analyze SSRF findings
    hunt_parser = subparsers.add_parser("hunt", help="Hunt for SSRF vulnerabilities using BBOT")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use (default: httpx,excavate,hunt)")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    # Analyze command - analyze findings from JSON file
    analyze_parser = subparsers.add_parser("analyze", help="Analyze SSRF findings from JSON file")
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