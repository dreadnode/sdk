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

dn.configure(project="cve-validator-agent")

console = Console()


@dn.task(name="Validate CVE", label="validate_cve", log_output=True)
async def validate_cve_finding(finding_data: dict) -> dict:
    """Validate a CVE finding for exploitability."""
    
    host = finding_data.get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    cve_id = extract_cve_from_description(description)
    
    console.print(f"[cyan]Validating CVE finding for: {host}[/cyan]")
    console.print(f"CVE: {cve_id}")
    console.print(f"Description: {description}")
    
    dn.log_input("finding_data", finding_data)
    dn.log_input("cve_id", cve_id)
    dn.log_input("target_host", host)
    
    try:
        agent = create_cve_validator_agent()
        
        analysis_task = f"""
        Validate the CVE {cve_id} on target {host}.
        
        IMPORTANT: You must first lookup the actual CVE details from authoritative sources.
        
        Your task:
        1. Use http_request to query CVE databases. Try these sources in order:
           - https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id} (NVD API)
           - https://cveawg.mitre.org/api/cve/{cve_id} (MITRE API)
           - https://cve.circl.lu/api/cve/{cve_id} (CIRCL API)
        2. Parse the JSON response to extract CVE description, affected software, and attack vectors
        3. Based on the ACTUAL CVE details, determine if this vulnerability applies to the target
        4. If applicable, use appropriate tools to probe the target for this specific vulnerability
        5. If exploitable, store the finding using store_cve_finding
        
        DO NOT rely on training data for CVE details - always lookup the CVE first using APIs.
        Be thorough in your testing and provide clear evidence for your conclusion.
        """
        
        result = await agent.run(analysis_task)
        
        tool_outputs = {}
        tools_used = []
        analysis_parts = []
        
        if hasattr(result, 'messages') and result.messages:
            for message in result.messages:
                if message.role == "assistant" and message.content:
                    analysis_parts.append(message.content)
                    console.print(f"[yellow]Agent analysis:[/yellow] {message.content}")
                elif message.role == "assistant" and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tools_used.append(tool_call.function.name)
                        console.print(f"[blue]Tool call:[/blue] {tool_call.function.name}")
                elif message.role == "tool":
                    tool_name = getattr(message, "name", "unknown")
                    tool_outputs[tool_name] = message.content
                    console.print(f"[green]Tool output from {tool_name}:[/green] {message.content[:200]}...")
                    dn.log_output(f"tool_output_{tool_name}", message.content)

        finding_stored = "store_cve_finding" in tools_used
        is_exploitable = finding_stored
        
        if result.messages and result.messages[-1].content:
            final_analysis = result.messages[-1].content.lower()
            is_exploitable = is_exploitable or any(
                phrase in final_analysis
                for phrase in [
                    "exploitable",
                    "vulnerable",
                    "confirmed",
                    "successfully exploited"
                ]
            )
        
        analysis_result = "\n\n".join(analysis_parts) if analysis_parts else "CVE validation completed"
        
        dn.log_metric("cve_validated", 1)
        dn.log_metric("tools_used", len(tools_used))
        dn.log_metric("is_exploitable", 1 if is_exploitable else 0)
        dn.log_metric("stored_in_db", 1 if finding_stored else 0)
        
        console.print(f"[green]Validation complete for {cve_id} on {host}[/green]")
        console.print(f"Exploitable: {is_exploitable}")
        console.print(f"Tools used: {len(tools_used)}")
        console.print(f"Stored in DB: {finding_stored}")
        
        return {
            "host": host,
            "cve_id": cve_id,
            "is_exploitable": is_exploitable,
            "analysis": analysis_result,
            "tools_used": tools_used,
            "tool_outputs": tool_outputs,
            "stored_in_db": finding_stored,
            "original_finding": finding_data,
            "timestamp": time.time()
        }
        
    except Exception as e:
        console.print(f"[red]Error validating CVE {cve_id} on {host}: {e}[/red]")
        return {"error": str(e), "host": host, "cve_id": cve_id}


def extract_cve_from_description(description: str) -> str:
    """Extract CVE ID from description."""
    import re
    cve_match = re.search(r'CVE-\d{4}-\d+', description)
    return cve_match.group(0) if cve_match else "UNKNOWN-CVE"


def create_cve_validator_agent() -> Agent:
    """Create a CVE validation agent."""
    tools = [KaliTool(), Neo4jTool(), BBotTool(), OastTool()]

    return Agent(
        name="cve-validator-agent",
        description="Validates CVE findings for exploitability",
        model="gpt-4-turbo",
        tools=tools,
        max_steps=10,
        instructions="""You are a security researcher specializing in CVE validation and exploitation.

Your job is to:
- Lookup actual CVE details from authoritative sources (NEVER rely on training data)
- Use available tools to probe targets for specific CVE vulnerabilities
- Determine if targets are exploitable based on evidence
- Store confirmed exploitable findings in Neo4j

Available tools:
- http_request: Make HTTP requests to CVE databases and target systems
- curl: Test endpoints with specific CVE payloads
- store_cve_finding: Store confirmed CVE exploits in Neo4j
- nmap: Port scanning and service detection
- dig_dns_lookup: DNS queries for reconnaissance

Be systematic in your approach:
1. FIRST: Use http_request to lookup CVE details from APIs:
   - https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-XXXX-XXXX (NVD API)
   - https://cveawg.mitre.org/api/cve/CVE-XXXX-XXXX (MITRE API) 
   - https://cve.circl.lu/api/cve/CVE-XXXX-XXXX (CIRCL API)
2. Parse JSON response to get accurate CVE description and affected software
3. Based on ACTUAL CVE data, determine if vulnerability applies to target
4. If applicable, craft appropriate test requests/payloads
5. Analyze responses for vulnerability indicators
6. Only mark as exploitable with clear evidence
7. Store findings when exploitation is confirmed

CRITICAL: Always lookup CVE details first using APIs - do not guess or use training data.""",
    )


async def validate_from_file(findings_file: Path) -> None:
    """Validate CVEs from a file of findings."""
    
    if not findings_file.exists():
        console.print(f"Error: File {findings_file} not found")
        return
        
    findings = []
    with findings_file.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    finding = json.loads(line)
                    findings.append(finding)
                except json.JSONDecodeError:
                    console.print(f"[yellow]Skipping invalid JSON line: {line[:50]}...[/yellow]")
    
    if not findings:
        console.print("No valid findings found in file")
        return
        
    console.print(f"Validating CVEs for {len(findings)} findings...")
    
    with dn.run("cve-validation-batch"):
        results = []
        for i, finding in enumerate(findings, 1):
            console.print(f"\n[{i}/{len(findings)}] Processing finding...")
            result = await validate_cve_finding(finding)
            results.append(result)
        
        successful = len([r for r in results if "error" not in r])
        exploitable = len([r for r in results if r.get("is_exploitable")])
        stored = len([r for r in results if r.get("stored_in_db")])
        
        dn.log_metric("total_findings", len(findings))
        dn.log_metric("successful_validations", successful) 
        dn.log_metric("exploitable_cves", exploitable)
        dn.log_metric("stored_findings", stored)
        
        console.print(f"\n[bold]Validation Summary:[/bold]")
        console.print(f"  Findings processed: {len(findings)}")
        console.print(f"  Successful validations: {successful}")
        console.print(f"  Exploitable CVEs: {exploitable}")
        console.print(f"  Stored in DB: {stored}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="CVE Validation Agent")
    subparsers = parser.add_subparsers(dest="command")

    # Validate from file
    file_parser = subparsers.add_parser("validate", help="Validate CVEs from findings file")
    file_parser.add_argument("findings_file", type=Path, help="JSON file containing findings (one per line)")

    # Single finding validation  
    single_parser = subparsers.add_parser("analyze", help="Analyze single finding")
    single_parser.add_argument("finding_json", help="JSON string of finding to analyze")

    args = parser.parse_args()

    if args.command == "validate":
        await validate_from_file(args.findings_file)
    elif args.command == "analyze":
        try:
            finding = json.loads(args.finding_json)
            with dn.run("single-cve-validation"):
                result = await validate_cve_finding(finding)
                if "error" not in result:
                    console.print(f"[green]✓[/green] Validation complete")
                else:
                    console.print(f"[red]✗[/red] {result['error']}")
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON provided[/red]")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())