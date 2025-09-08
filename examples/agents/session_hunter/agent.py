import argparse
import asyncio
import json
import typing as t
from pathlib import Path

from rich.console import Console

import dreadnode as dn
from dreadnode.agent.agent import Agent
from dreadnode.agent.result import AgentResult
from dreadnode.agent.tools.bbot.tool import BBotTool
from dreadnode.agent.tools.kali.tool import KaliTool
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

dn.configure(server=None, token=None, project="crypto-hunter-agent", console=False)

console = Console()


@dn.task(name="Analyze Crypto Finding", label="analyze_crypto_finding")
async def analyze_crypto_finding(crypto_event: dict[str, t.Any]) -> dict[str, t.Any]:
    """Analyze cryptographic product finding using autonomous agent."""
    
    description = crypto_event.get("data", {}).get("description", "")
    url = crypto_event.get("data", {}).get("url", "")
    host = crypto_event.get("host", "")
    
    console.print(f"[cyan]Crypto Hunter analyzing cryptographic finding...[/cyan]")
    console.print(f"Host: {host}")
    console.print(f"URL: {url}")
    console.print(f"Finding: {description}")
    
    # Create autonomous crypto hunter agent
    try:
        agent = create_crypto_hunter_agent()
    except Exception as e:
        console.print(f"[red]Error creating agent: {e}[/red]")
        return {
            "error": f"Agent creation failed: {e}",
            "event_id": crypto_event.get("id"),
            "host": host,
            "url": url
        }
    
    # Let the agent autonomously analyze the cryptographic vulnerability
    analysis_task = f"""
    I've discovered a cryptographic security finding on {host}:
    
    URL: {url}
    Finding Description: {description}
    
    This could involve session cookies, JWT tokens, API keys, encryption keys, or other cryptographic products. 
    
    You MUST use the available tools to perform hands-on analysis:
    
    1. Identify the specific cryptographic product type and implementation
    2. Use http_request tool to extract and analyze cryptographic artifacts (cookies, tokens, keys, etc.)  
    3. Use curl tool to test different endpoints and gather more data
    4. Use hashcat tool if you find hashes or secrets to crack
    5. Test for common cryptographic weaknesses using your tools
    6. Develop proof-of-concept exploits demonstrating the security impact
    7. Assess the full scope of potential compromise
        
    START by immediately using the http_request or curl tools to gather data from the finding URL. Don't just analyze theoretically - actively investigate using tools!
    """
    
    dn.log_input("system_prompt_and_event", {
        "system_prompt": analysis_task,
        "event": crypto_event
    })
    
    try:
        result = await agent.run(analysis_task)
        
        console.print("="*80)
        
        for i, message in enumerate(result.messages, 1):
            role_color = {
                "system": "dim",
                "user": "green", 
                "assistant": "cyan",
                "tool": "yellow"
            }.get(message.role, "white")
            
            console.print(f"\n[{role_color}]Message {i} ({message.role.upper()}):[/{role_color}]")
            
            if message.role == "assistant" and message.tool_calls:
                console.print(f"[{role_color}]Agent Response:[/{role_color}] {message.content}")
                console.print(f"[bold {role_color}]Tool Calls:[/bold {role_color}]")
                for tool_call in message.tool_calls:
                    console.print(f"  - {tool_call.function.name}({tool_call.function.arguments})")
            elif message.role == "tool":
                tool_name = getattr(message, "name", "unknown")
                console.print(f"[{role_color}]Tool '{tool_name}' Output:[/{role_color}]")
                console.print(f"  {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
            else:
                if message.role == "assistant":
                    console.print(f"[{role_color}]{message.content or 'No content'}[/{role_color}]")
                else:
                    content_preview = message.content[:300] if message.content else "No content"
                    console.print(f"[{role_color}]{content_preview}{'...' if len(message.content) > 300 else ''}[/{role_color}]")
        
        console.print("\n" + "="*80)
        console.print(f"[bold]Agent Steps:[/bold] {result.steps}")
        console.print(f"[bold]Token Usage:[/bold] {result.usage}")
        
        # Capture the full agent analysis from all assistant messages
        analysis_parts = []
        for message in result.messages:
            if message.role == "assistant" and message.content:
                analysis_parts.append(message.content)
        
        analysis_result = "\n\n".join(analysis_parts) if analysis_parts else "No analysis provided"
        
        verdict = extract_verdict_from_analysis(analysis_result, crypto_event)
        
        # Log metrics (now within task context)
        dn.log_metric("crypto_finding_analyzed", 1)
        if "critical" in analysis_result.lower() or "vulnerable" in analysis_result.lower():
            dn.log_metric("critical_vulnerabilities", 1)
        
        # Extract tool usage like other agents
        tool_outputs = {}
        tools_used = []
        
        for message in result.messages:
            if message.role == "assistant" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tools_used.append(tool_name)
            elif message.role == "tool":
                tool_name = getattr(message, "name", "unknown")
                tool_outputs[tool_name] = message.content[:200] + "..." if len(message.content) > 200 else message.content
        
        final_result = {
            "event_id": crypto_event.get("id"),
            "host": host,
            "url": url,
            "has_vulnerability": verdict.get("vulnerable", False),
            "severity": verdict.get("severity", "unknown"),
            "key_findings": verdict.get("key_findings", []),
            "agent_analysis": analysis_result,
            "verdict": verdict,
            "tools_used": tools_used,
            "tool_outputs": tool_outputs,
            "original_event": crypto_event,
            "timestamp": crypto_event.get("timestamp")
        }
        
        console.print(f"\n[bold green]AUTONOMOUS ANALYSIS COMPLETE[/bold green]")
        console.print(f"[bold]Final Verdict:[/bold] [{'red' if verdict.get('severity') == 'critical' else 'yellow'}]{verdict.get('severity', 'unknown').upper()}[/]")
        console.print(f"[bold]Vulnerable:[/bold] {verdict.get('vulnerable')}")
        console.print(f"[bold]Key Findings:[/bold] {', '.join(verdict.get('key_findings', []))}")
        console.print(f"[bold]Assessment Type:[/bold] {verdict.get('finding_type')}")
        
        dn.log_output("analysis", analysis_result)
        
        return final_result
        
    except Exception as e:
        console.print(f"[red]Error in autonomous analysis: {e}[/red]")
        return {
            "error": f"Agent analysis failed: {e}",
            "event_id": crypto_event.get("id"),
            "host": host,
            "url": url
        }


def extract_verdict_from_analysis(analysis: str, event: dict) -> dict[str, t.Any]:
    """Extract structured verdict from agent's autonomous analysis."""
    
    analysis_lower = analysis.lower()
    
    severity = "low"
    vulnerable = False
    
    if any(word in analysis_lower for word in ["critical", "severe", "high risk", "exploitable"]):
        severity = "critical"
        vulnerable = True
    elif any(word in analysis_lower for word in ["medium", "moderate", "concerning"]):
        severity = "medium"
        vulnerable = "vulnerable" in analysis_lower or "exploitable" in analysis_lower
    elif any(word in analysis_lower for word in ["low", "minor", "informational"]):
        severity = "low"
    
    findings = []
    if "secret" in analysis_lower and ("cracked" in analysis_lower or "discovered" in analysis_lower):
        findings.append("Session secret compromised")
    if "forge" in analysis_lower or "craft" in analysis_lower:
        findings.append("Session forgery possible")
    if "bypass" in analysis_lower and "auth" in analysis_lower:
        findings.append("Authentication bypass possible")
    
    return {
        "finding_type": "cryptographic_analysis",
        "host": event.get("host"),
        "url": event.get("data", {}).get("url"),
        "severity": severity,
        "vulnerable": vulnerable,
        "autonomous_analysis": True,
        "key_findings": findings,
        "agent_assessment": analysis,
        "timestamp": event.get("timestamp")
    }

def create_crypto_hunter_agent() -> Agent:
    """Create an autonomous crypto hunter agent with full tool access."""
    tools = [BBotTool(), KaliTool(), OastTool()]

    return Agent(
        name="crypto-hunter-agent",
        description="Autonomous agent for analyzing cryptographic vulnerabilities across all implementations",
        model="gpt-4-turbo",
        tools=tools,
        max_steps=10,
        instructions="""You are an elite cryptographic security researcher with expertise across all cryptographic implementations and protocols.

You have complete autonomy to analyze ANY type of cryptographic vulnerability using creative methods and all available tools. Be innovative, thorough, and think like an advanced cryptographic attacker.

CRYPTOGRAPHIC EXPERTISE AREAS:
- Session management (Express.js, Django, Rails, PHP, .NET, etc.)
- JSON Web Tokens (JWT) and OAuth implementations  
- API keys and authentication tokens
- Encryption keys and cipher implementations
- Certificate and PKI vulnerabilities
- Cryptographic libraries and frameworks
- Custom crypto implementations

CAPABILITIES & APPROACH:
- Use BBotTool for additional reconnaissance and cryptographic artifact discovery
- Use KaliTool for hashcat password cracking, HTTP testing, and cryptanalysis
- Use OastTool for out-of-band testing and callback verification
- Create your own cryptanalysis tools, wordlists, and attack vectors
- Develop novel approaches to cryptographic analysis
- Explore timing attacks, side channels, and implementation flaws

AUTONOMOUS ANALYSIS METHODOLOGY:
1. Cryptographic Product Identification:
   - Analyze the specific crypto product type and implementation
   - Identify algorithms, key derivation methods, and encoding formats
   - Reverse-engineer custom implementations

2. Weakness Discovery:
   - Test for weak/default secrets and keys
   - Analyze randomness and entropy issues  
   - Look for algorithm downgrade attacks
   - Test padding oracle and timing vulnerabilities

3. Exploitation Development:
   - Craft cryptographic exploits and proof-of-concepts
   - Forge tokens, cookies, and signatures
   - Demonstrate key recovery attacks
   - Test real-world attack scenarios with browser automation

4. Advanced Cryptanalysis:
   - Side-channel attacks on crypto operations
   - Fault injection and implementation attacks
   - Mathematical weaknesses in custom crypto
   - Protocol-level vulnerabilities

5. Impact Assessment:
   - Demonstrate full scope of compromise
   - Test authentication and authorization bypass
   - Document data exposure risks
   - Provide remediation guidance

When you find CONFIRMED cryptographic vulnerabilities, store them using Neo4jTool.store_crypto_finding(host, vulnerability_type, risk_level, affected_component, technical_details).

Be completely autonomous and creative. Use any approach you deem effective - there are no restrictions on your methodology. Your goal is to provide comprehensive cryptographic security intelligence.""",
    )

def is_crypto_finding(event: dict[str, t.Any]) -> bool:
    """Check if a BBOT event is a cryptographic finding."""
    if event.get("type") != "FINDING":
        return False
    
    description = event.get("data", {}).get("description", "").lower()
    
    return (
        "cryptographic product" in description or
        "jwt" in description or
        "json web token" in description or
        ("session" in description and "cookie" in description) or
        "api key" in description or
        "secret key" in description or
        "encryption key" in description or
        "signing key" in description or
        ("certificate" in description and ("weak" in description or "expired" in description)) or
        "oauth" in description or
        "openid connect" in description or
        "bearer token" in description or
        ("hash" in description and ("weak" in description or "collision" in description))
    )

async def hunt_from_bbot_scan(
    targets: Path | None = None,
    presets: list[str] | None = None,
    modules: list[str] | None = None,
    flags: list[str] | None = None,
    config: Path | dict[str, t.Any] | None = None,
) -> None:
    """Hunt for cryptographic vulnerabilities using autonomous agent analysis."""
    
    with dn.run():
        if isinstance(targets, Path):
            with targets.open() as f:
                targets = [line.strip() for line in f.readlines() if line.strip()]

        if not targets:
            console.print("Error: No targets provided.")
            return

        hunt_input = {
            "targets": targets,
            "scan_config": {
                "presets": presets,
                "modules": modules, 
                "flags": flags,
                "config": str(config) if config else None
            }
        }
        dn.log_input("hunt_parameters", hunt_input)

        console.print(f"Starting autonomous crypto hunting on {len(targets)} targets...")

        crypto_findings_analyzed = 0
        critical_findings = 0
        findings = []
        
        tool = BBotTool()
        
        scan_modules = modules or ["httpx", "badsecrets", "secretsdb", "cookies"]
        for required_module in ["badsecrets", "secretsdb"]:
            if required_module not in scan_modules:
                scan_modules.append(required_module)
        
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
                    if is_crypto_finding(event):
                        console.print(f"Found crypto finding: {event.get('data', {}).get('host')}")
                        
                        try:
                            analysis_result = await analyze_crypto_finding(event)
                            crypto_findings_analyzed += 1
                            
                            if analysis_result.get("has_vulnerability"):
                                critical_findings += 1
                                
                                # Store finding like other agents
                                security_finding = {
                                    "url": analysis_result["url"],
                                    "host": analysis_result["host"],
                                    "finding_type": "crypto_vulnerability",
                                    "severity": analysis_result["severity"],
                                    "key_findings": analysis_result["key_findings"],
                                    "analysis": analysis_result["agent_analysis"],
                                    "tool_outputs": analysis_result["tool_outputs"],
                                    "timestamp": analysis_result["timestamp"],
                                }
                                
                                findings.append(security_finding)
                                dn.log_output(f"crypto_finding_{analysis_result['host']}", security_finding)
                                console.print(f"[green]Crypto vulnerability confirmed on {analysis_result['host']}[/green]")
                            else:
                                console.print(f"No exploitable crypto vulnerability on {event.get('host')}")
                            
                        except Exception as e:
                            console.print(f"Error in autonomous crypto analysis: {e}")

            except Exception as e:
                console.print(f"Error scanning {target}: {e}")
            
        dn.log_metric("crypto_findings_analyzed", crypto_findings_analyzed)
        dn.log_metric("critical_vulnerabilities", critical_findings)
        if findings:
            dn.log_output("findings", findings)
        
        console.print(f"Hunt Summary:")
        console.print(f"   Crypto findings analyzed: {crypto_findings_analyzed}")
        console.print(f"   Critical vulnerabilities: {critical_findings}")

async def analyze_finding_file(finding_file: Path) -> None:
    """Analyze BBOT findings from JSON/JSONL file using autonomous agent."""
    
    with dn.run():
        dn.log_input("finding_file", str(finding_file))
        
        console.print(f"Analyzing finding file: {finding_file}")
        
        crypto_findings = []
        total_events = 0
        
        try:
            with finding_file.open() as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        finding_data = json.loads(line)
                        total_events += 1
                        
                        if is_crypto_finding(finding_data):
                            crypto_findings.append(finding_data)
                            
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]Skipping invalid JSON on line {line_num}: {e}[/yellow]")
                        continue
            
            console.print(f"Found {len(crypto_findings)} cryptographic findings out of {total_events} total events")
            
            if not crypto_findings:
                console.print("No cryptographic findings detected in this file.")
                return
            
            for i, finding_data in enumerate(crypto_findings, 1):
                console.print(f"\n[cyan]--- Analyzing Crypto Finding {i}/{len(crypto_findings)} ---[/cyan]")
                
                analysis_result = await analyze_crypto_finding(finding_data)
                
                console.print(f"\n[bold]Autonomous Crypto Analysis Results:[/bold]")
                verdict = analysis_result.get("verdict", {})
                console.print(f"Host: {verdict.get('host')}")
                console.print(f"URL: {verdict.get('url')}")
                console.print(f"Severity: {verdict.get('severity', 'unknown').upper()}")
                console.print(f"Vulnerable: {verdict.get('vulnerable')}")
                console.print(f"Key Findings: {', '.join(verdict.get('key_findings', []))}")
                
                if i < len(crypto_findings):
                    console.print("\n" + "="*50)
            
            dn.log_metric("total_events", total_events)  
            dn.log_metric("crypto_findings_found", len(crypto_findings))
            
        except Exception as e:
            console.print(f"Error analyzing finding file: {e}")
            dn.log_output("error", str(e))

async def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Hunter - Autonomous cryptographic vulnerability analyzer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hunt_parser = subparsers.add_parser("hunt", help="Hunt for crypto vulnerabilities using autonomous agent")
    hunt_parser.add_argument("--targets", type=Path, help="Path to file containing targets")
    hunt_parser.add_argument("--presets", nargs="*", help="BBOT presets to use")
    hunt_parser.add_argument("--modules", nargs="*", help="BBOT modules to use")
    hunt_parser.add_argument("--flags", nargs="*", help="BBOT flags to use")
    hunt_parser.add_argument("--config", type=Path, help="Path to config file")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single BBOT finding with autonomous agent")
    analyze_parser.add_argument("finding_file", type=Path, help="Path to BBOT finding JSON file")

    args = parser.parse_args()

    if args.command == "hunt":
        await hunt_from_bbot_scan(args.targets, args.presets, args.modules, args.flags, args.config)
    elif args.command == "analyze":
        await analyze_finding_file(args.finding_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())