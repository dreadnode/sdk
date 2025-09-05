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

dn.configure(server=None, token=None, project="error-based-sqli-agent", console=False)

console = Console()

@dn.task(name="Analyze Error-Based SQLi Finding", label="analyze_error_sqli_finding")
async def analyze_sqli_finding(finding_data: dict[str, t.Any]) -> dict[str, t.Any]:
    """Analyze a BBOT SQL injection finding for error-based exploitability."""
    sqli_agent = create_sqli_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    param_name = extract_param_name(description)
    param_type = extract_param_type(description)
    original_value = extract_original_value(description)
    
    result = await sqli_agent.run(
        f"Analyze the potential SQL injection vulnerability at {url} using parameter '{param_name}'. "
        f"The original parameter value was: {original_value}\n\n"
        f"Focus on ERROR-BASED SQL injection techniques. Test for database errors that leak information. "
        f"Start simple and adapt based on error messages you receive."
    )

    tool_outputs = {}
    tools_used = []
    
    for message in result.messages:
        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tools_used.append(tool_name)
        elif message.role == "tool":
            tool_name = getattr(message, "name", "unknown")
            tool_outputs[tool_name] = message.content

    finding_stored = "store_sqli_finding" in tools_used
    has_sqli = finding_stored
    if result.messages and result.messages[-1].content:
        has_sqli = has_sqli or any(
            phrase in result.messages[-1].content.lower()
            for phrase in [
                "error-based injection confirmed",
                "database error detected",
                "mysql error",
                "postgresql error", 
                "oracle error",
                "mssql error",
                "sql syntax error",
                "database version extracted",
                "error message leaked",
            ]
        )

    dn.log_metric("tools_used", len(tools_used))
    dn.log_metric("has_error_sqli", 1 if has_sqli else 0)
    dn.log_metric("stored_in_db", 1 if finding_stored else 0)
    
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
        "original_finding": finding_data
    }

    return analysis_result

def create_sqli_agent() -> Agent:
    """Create an error-based SQL injection analysis agent."""
    tools = [BBotTool(), KaliTool(), Neo4jTool()]

    return Agent(
        name="error-based-sqli-agent",
        description="An agent that analyzes and exploits error-based SQL injection vulnerabilities",
        model="gpt-4-turbo",
        tools=tools,
        instructions="""You are an expert at analyzing and exploiting error-based SQL injection vulnerabilities.

Your mission is to trigger verbose database errors that leak information. Start simple and adapt based on what you discover:

1. Begin with basic error triggers (single quotes, double quotes, parentheses)
2. Analyze error messages to identify the database type and technology stack
3. Based on the database you identify, craft targeted error-based payloads
4. Extract information progressively through error message analysis
5. Document findings and iterate your approach based on responses

Use the http_request tool to test payloads systematically. Let the application's responses guide your next steps.

If you confirm error-based SQL injection exists, use store_sqli_finding to record the vulnerability.""",
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

def is_sqli_finding(event: dict[str, t.Any]) -> bool:
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
    """Hunt for error-based SQL injection vulnerabilities from BBOT scan findings."""
    
    if isinstance(targets, Path):
        with targets.open() as f:
            targets = [line.strip() for line in f.readlines() if line.strip()]

    if not targets:
        console.print("Error: No targets provided.")
        return

    with dn.run("error-based-sqli-hunt"):
        dn.log_params(
            target_count=len(targets),
            presets=presets or [],
            modules=modules or [],
            flags=flags or [],
        )

        console.print(f"Starting error-based SQL injection hunt on {len(targets)} targets...")

        sqli_findings_count = 0
        total_findings = 0
        
        tool = BBotTool()
        scan_modules = modules or ["httpx", "excavate", "hunt"]
        
        for target in targets:
            try:
                scan_config = config or {"omit_event_types": []}
                
                events = tool.run(
                    target=target,
                    presets=presets,
                    modules=scan_modules,
                    flags=flags,
                    config=scan_config,
                )
                
                async for event in events:
                    if is_sqli_finding(event):
                        total_findings += 1
                        
                        try:
                            analysis_result = await analyze_sqli_finding(event)
                            
                            if analysis_result["has_sqli"]:
                                sqli_findings_count += 1
                                
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"], 
                                    "parameter": analysis_result["parameter"],
                                    "finding_type": "error_based_sqli",
                                    "risk_level": "high",
                                    "analysis": analysis_result["analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": time.time(),
                                    "stored_in_db": analysis_result["stored_in_db"],
                                }
                                
                                dn.log_output(f"error_sqli_finding_{analysis_result['host']}", security_finding)
                                console.print(f"Error-based SQL injection confirmed on {analysis_result['host']}")
                            else:
                                console.print(f"Error-based SQL injection not exploitable on {event.get('host')}")
                                
                        except Exception as e:
                            console.print(f"Error analyzing SQL injection finding: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")

        dn.log_metric("total_findings", total_findings)
        dn.log_metric("error_based_confirmed", sqli_findings_count)
        
        console.print(f"Hunt Summary:")
        console.print(f"   SQL injection candidates found: {total_findings}")
        console.print(f"   Error-based SQL injection vulnerabilities confirmed: {sqli_findings_count}")

async def analyze_finding_file(finding_file: Path, debug: bool = False) -> None:
    """Analyze SQL injection findings from a JSON file."""
    
    with dn.run("error-based-sqli-analyze"):        
        try:
            with finding_file.open() as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            sqli_count = 0
            for finding in findings:
                if is_sqli_finding(finding):
                    analysis_result = await analyze_sqli_finding(finding)
                    
                    if debug:
                        console.print(f"Tools used: {', '.join(analysis_result['tools_used'])}")
                    
                    if analysis_result["has_sqli"]:
                        sqli_count += 1
                        console.print(f"Error-based SQL injection confirmed")
                    else:
                        console.print(f"No error-based SQL injection exploitation possible")
            
            dn.log_metric("error_based_findings", sqli_count)
            
        except Exception as e:
            console.print(f"Error analyzing findings file: {e}")

async def main() -> None:
    parser = argparse.ArgumentParser(description="Error-based SQL injection vulnerability hunter")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hunt_parser = subparsers.add_parser("hunt", help="Hunt for error-based SQL injection vulnerabilities using BBOT")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use")
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