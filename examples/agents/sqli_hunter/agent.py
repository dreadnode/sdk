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

dn.configure(server=None, token=None, project="sqli-hunter-agent", console=False)

console = Console()


@dn.task(name="Analyze SQLi Finding", label="analyze_sqli_finding")
async def analyze_sqli_finding(finding_data: dict) -> dict:
    """Analyze a BBOT SQL injection finding for exploitability."""
    sqli_agent = create_sqli_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    param_name = extract_param_name(description)
    param_type = extract_param_type(description)
    original_value = extract_original_value(description)
    
    console.print(f"[*] Analyzing SQL injection finding on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Parameter: {param_name} ({param_type})")
    console.print(f"    Original value: {original_value}")
    
    result = await sqli_agent.run(
        f"Analyze the potential SQL injection vulnerability at {url} using parameter '{param_name}'. "
        f"The original parameter value was: {original_value}\n\n"
        f"Use the tools available to you to test for SQL injection by modifying the {param_name} parameter. "
        f"Test systematically using your expertise to choose appropriate payloads and analyze responses."
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

    finding_stored = "store_sqli_finding" in tools_used
    has_sqli = finding_stored
    if result.messages and result.messages[-1].content:
        has_sqli = has_sqli or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "sql injection confirmed",
                "sqli confirmed",
                "injection successful",
                "database error",
                "syntax error",
                "mysql error",
                "postgresql error",
                "oracle error",
                "mssql error",
                "union select",
                "sql vulnerability confirmed",
                "injection point found",
                "database accessible",
                "sql injection detected"
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_sqli", 1 if has_sqli else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    dn.log_output("raw_tool_data", tool_outputs)
    
    db_version = "See agent analysis for database details" if has_sqli else ""
    
    analysis_result = {
        "url": url,
        "host": host,
        "parameter": param_name,
        "param_type": param_type,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "analysis": result.messages[-1].content if result.messages else None,
        "steps": result.steps,
        "has_sqli": has_sqli,
        "stored_in_db": finding_stored,
        "db_version": db_version,
        "original_finding": finding_data
    }

    return analysis_result


def create_sqli_agent() -> Agent:
    """Create a SQL injection vulnerability analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool()]

    return Agent(
        name="sqli-hunter-agent",
        description="An agent that analyzes and exploits SQL injection vulnerabilities",
        model="gpt-4-turbo",
        tools=tools,
        instructions="""You are an expert at analyzing and exploiting SQL injection vulnerabilities.

Use the tools available to you to test for SQL injection by modifying the vulnerable parameter with various payloads.

Analyze responses for indicators of SQL injection. If you confirm SQL injection exists, attempt to identify the database type and version through error messages, version queries, or response patterns.

Be thorough and systematic in your testing approach. Use your expertise to choose appropriate payloads and techniques.

If you confirm SQL injection exists, use store_sqli_finding to record the vulnerability with details about the successful payload and evidence.""",
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
        end = description.rfind("]")
        return description[start:end] if end > start else ""
    return ""



def is_sqli_finding(event: dict) -> bool:
    """Check if a BBOT event is a SQL injection finding."""
    if event.get('type') != 'FINDING':
        return False
    
    description = event.get('data', {}).get('description', '')
    return 'SQL Injection' in description


async def hunt_from_bbot_scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for SQL injection vulnerabilities from BBOT scan findings."""
    
    if isinstance(targets, Path):
        with Path.open(targets) as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided. Use --targets to specify targets.")
        return

    with dn.run("sqli-hunt-from-bbot"):
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
        )

        console.print(f"Starting SQL injection hunt on {len(targets)} targets using BBOT scan...")

        sqli_findings_count = 0
        total_findings = 0
        
        tool = BBotTool()
        
        scan_modules = modules or ["httpx", "excavate", "hunt"]
        
        for target in targets:
            try:
                console.print(f"[*] Scanning {target} for SQL injection parameters...")
                
                events = tool.run(
                    target=target,
                    presets=presets,
                    modules=scan_modules,
                    flags=flags,
                    config=config,
                )
                
                async for event in events:
                    if is_sqli_finding(event):
                        total_findings += 1
                        console.print(f"Found SQL injection candidate on {event.get('host')}")
                        
                        try:
                            analysis_result = await analyze_sqli_finding(event)
                            
                            if analysis_result["has_sqli"]:
                                sqli_findings_count += 1
                                
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"], 
                                    "parameter": analysis_result["parameter"],
                                    "finding_type": "sqli",
                                    "risk_level": "high",
                                    "analysis": analysis_result["analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": time.time(),
                                    "stored_in_db": analysis_result["stored_in_db"],
                                    "db_version": analysis_result["db_version"],
                                }
                                
                                dn.log_output(f"sqli_finding_{analysis_result['host']}", security_finding)
                                console.print(f"SQL INJECTION CONFIRMED on {analysis_result['host']}")
                            else:
                                console.print(f"SQL injection not exploitable on {event.get('host')}")
                                
                        except Exception as e:
                            console.print(f"Error analyzing SQL injection finding: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")

        dn.log_metric("total_findings", total_findings)
        dn.log_metric("sqli_confirmed", sqli_findings_count)
        
        console.print(f"\nHunt Summary:")
        console.print(f"   SQL injection candidates found: {total_findings}")
        console.print(f"   SQL injection vulnerabilities confirmed: {sqli_findings_count}")


async def analyze_finding_file(finding_file: Path, debug: bool = False) -> None:
    """Analyze SQL injection findings from a JSON file (for testing)."""
    
    with dn.run("sqli-analyze-findings"):
        console.print(f"Analyzing findings from {finding_file}")
        
        try:
            with open(finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            sqli_count = 0
            for finding in findings:
                if is_sqli_finding(finding):
                    console.print(f"[*] Analyzing SQL injection finding...")
                    analysis_result = await analyze_sqli_finding(finding)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                        console.print(f"Analysis: {analysis_result['analysis'][:200]}...")
                    
                    if analysis_result["has_sqli"]:
                        sqli_count += 1
                        console.print(f"SQL INJECTION CONFIRMED!")
                    else:
                        console.print(f"No SQL injection exploitation possible")
            
            dn.log_metric("sqli_findings", sqli_count)
            console.print(f"\nAnalysis Summary:")
            console.print(f"   SQL injection vulnerabilities confirmed: {sqli_count}")
            
        except Exception as e:
            console.print(f"Error analyzing findings file: {e}")


async def main():
    parser = argparse.ArgumentParser(description="SQL injection vulnerability hunter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hunt_parser = subparsers.add_parser("hunt", help="Hunt for SQL injection vulnerabilities using BBOT")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use (default: httpx,excavate,hunt)")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze SQL injection findings from JSON file")
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