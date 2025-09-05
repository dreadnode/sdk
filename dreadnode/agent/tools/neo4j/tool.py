import os
import typing as t
from datetime import datetime

from loguru import logger

from dreadnode.agent.tools import Toolset, tool_method

from neo4j import AsyncGraphDatabase, AsyncDriver


class Neo4jTool(Toolset):
    """Neo4j database tool for storing and querying security findings."""
    
    tool_name: str = "neo4j-tool"
    description: str = "Store and query security findings in Neo4j graph database"
    
    async def _get_driver(self) -> AsyncDriver:
        """Get or create Neo4j driver."""
        if not hasattr(self, '_driver') or not self._driver:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
            try:
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
        return self._driver
    
    @tool_method()
    async def store_subdomain_takeover_finding(
        self,
        subdomain: str,
        vulnerability_type: str,
        risk_level: str,
        cname_target: str | None = None,
        error_message: str | None = None,
        service_provider: str | None = None,
    ) -> str:
        """Store a confirmed subdomain takeover vulnerability finding.
        
        Args:
            subdomain: The vulnerable subdomain
            vulnerability_type: Type of vulnerability (e.g., "subdomain_takeover")
            risk_level: Risk level (HIGH, MEDIUM, LOW)
            cname_target: CNAME target if applicable
            error_message: Error message indicating unclaimed resource
            service_provider: Third-party service provider (AWS, GitHub, etc.)
            
        Returns:
            Confirmation message
        """
        try:
            driver = await self._get_driver()
            
            query = """
            MERGE (s:Subdomain {name: $subdomain})
            MERGE (f:Finding {
                id: $finding_id,
                subdomain: $subdomain,
                type: $vulnerability_type,
                risk_level: $risk_level,
                discovered_at: $timestamp
            })
            MERGE (s)-[:HAS_FINDING]->(f)
            SET f.cname_target = $cname_target,
                f.error_message = $error_message,  
                f.service_provider = $service_provider,
                f.updated_at = $timestamp
            RETURN f.id as finding_id
            """
            
            params = {
                "subdomain": subdomain,
                "finding_id": f"{subdomain}_{vulnerability_type}_{datetime.now().isoformat()}",
                "vulnerability_type": vulnerability_type,
                "risk_level": risk_level.upper(),
                "timestamp": datetime.now().isoformat(),
                "cname_target": cname_target,
                "error_message": error_message,
                "service_provider": service_provider,
            }
            
            async with driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
            logger.info(f"Stored subdomain takeover finding for {subdomain}")
            return f"Successfully stored finding: {record['finding_id'] if record else 'unknown'}"
            
        except Exception as e:
            logger.error(f"Failed to store finding: {e}")
            return f"Failed to store finding: {e}"

    @tool_method()
    async def store_ssrf_finding(
        self,
        url: str,
        parameter: str,
        vulnerability_type: str,
        risk_level: str,
        payload: str,
        response_evidence: str,
        internal_service_accessed: str = "",
    ) -> str:
        """Store an SSRF vulnerability finding in Neo4j.
        
        Args:
            url: Target URL with vulnerable parameter
            parameter: Parameter name that's vulnerable
            vulnerability_type: Type of SSRF (e.g., "blind_ssrf", "full_response_ssrf")
            risk_level: Risk level (low/medium/high/critical)
            payload: Successful SSRF payload used
            response_evidence: Evidence of SSRF (response differences, error messages)
            internal_service_accessed: Internal service that was accessed (if any)
            
        Returns:
            Confirmation message with finding ID
        """
        driver = await self._get_driver()
        
        try:
            query = """
            CREATE (f:SSRFVulnerability {
                url: $url,
                parameter: $parameter,
                vulnerability_type: $vulnerability_type,
                risk_level: $risk_level,
                timestamp: $timestamp,
                payload: $payload,
                response_evidence: $response_evidence,
                internal_service_accessed: $internal_service_accessed,
                finding_id: randomUUID()
            })
            RETURN f.finding_id as finding_id
            """
            
            params = {
                "url": url,
                "parameter": parameter,
                "vulnerability_type": vulnerability_type,
                "risk_level": risk_level.upper(),
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "response_evidence": response_evidence,
                "internal_service_accessed": internal_service_accessed,
            }
            
            async with driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
            logger.info(f"Stored SSRF finding for {url}")
            return f"Successfully stored SSRF finding: {record['finding_id'] if record else 'unknown'}"
            
        except Exception as e:
            logger.error(f"Failed to store SSRF finding: {e}")
            return f"Failed to store SSRF finding: {e}"

    @tool_method()
    async def store_sqli_finding(
        self,
        url: str,
        parameter: str,
        vulnerability_type: str,
        risk_level: str,
        payload: str,
        response_evidence: str,
        database_type: str = "",
        injection_point: str = "",
    ) -> str:
        """Store a SQL injection vulnerability finding in Neo4j.
        
        Args:
            url: Target URL with vulnerable parameter
            parameter: Parameter name that's vulnerable
            vulnerability_type: Type of SQLi (e.g., "union_based", "boolean_blind", "time_based", "error_based")
            risk_level: Risk level (low/medium/high/critical)
            payload: Successful SQL injection payload used
            response_evidence: Evidence of SQLi (error messages, data extraction, timing differences)
            database_type: Detected database type (MySQL, PostgreSQL, MSSQL, etc.)
            injection_point: Where injection occurs (GET, POST, Cookie, etc.)
            
        Returns:
            Confirmation message with finding ID
        """
        driver = await self._get_driver()
        
        try:
            query = """
            CREATE (f:SQLInjectionVulnerability {
                url: $url,
                parameter: $parameter,
                vulnerability_type: $vulnerability_type,
                risk_level: $risk_level,
                timestamp: $timestamp,
                payload: $payload,
                response_evidence: $response_evidence,
                database_type: $database_type,
                injection_point: $injection_point,
                finding_id: randomUUID()
            })
            RETURN f.finding_id as finding_id
            """
            
            params = {
                "url": url,
                "parameter": parameter,
                "vulnerability_type": vulnerability_type,
                "risk_level": risk_level.upper(),
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "response_evidence": response_evidence,
                "database_type": database_type,
                "injection_point": injection_point,
            }
            
            async with driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
            logger.info(f"Stored SQL injection finding for {url}")
            return f"Successfully stored SQL injection finding: {record['finding_id'] if record else 'unknown'}"
            
        except Exception as e:
            logger.error(f"Failed to store SQL injection finding: {e}")
            return f"Failed to store SQL injection finding: {e}"

    @tool_method()
    async def store_host_header_finding(
        self,
        url: str,
        vulnerability_type: str,
        risk_level: str,
        payload: str,
        response_evidence: str,
        reflection_location: str = "",
        impact: str = "",
    ) -> str:
        """Store a Host Header Injection vulnerability finding in Neo4j.
        
        Args:
            url: Target URL with vulnerable host header
            vulnerability_type: Type of host header injection (e.g., "reflection", "cache_poisoning", "password_reset")
            risk_level: Risk level (low/medium/high/critical)
            payload: Malicious host header value used
            response_evidence: Evidence of injection (reflected content, headers, etc.)
            reflection_location: Where the host header is reflected (headers, body, etc.)
            impact: Potential impact description
            
        Returns:
            Confirmation message with finding ID
        """
        driver = await self._get_driver()
        
        try:
            query = """
            CREATE (f:HostHeaderInjectionVulnerability {
                url: $url,
                vulnerability_type: $vulnerability_type,
                risk_level: $risk_level,
                timestamp: $timestamp,
                payload: $payload,
                response_evidence: $response_evidence,
                reflection_location: $reflection_location,
                impact: $impact,
                finding_id: randomUUID()
            })
            RETURN f.finding_id as finding_id
            """
            
            params = {
                "url": url,
                "vulnerability_type": vulnerability_type,
                "risk_level": risk_level.upper(),
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "response_evidence": response_evidence,
                "reflection_location": reflection_location,
                "impact": impact,
            }
            
            async with driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
            logger.info(f"Stored host header injection finding for {url}")
            return f"Successfully stored host header injection finding: {record['finding_id'] if record else 'unknown'}"
            
        except Exception as e:
            logger.error(f"Failed to store host header injection finding: {e}")
            return f"Failed to store host header injection finding: {e}"
    
    @tool_method()
    async def store_graphql_endpoint(
        self,
        url: str,
        schema_info: str,
        types_data: str = "",
        queries_data: str = "",
        mutations_data: str = "",
        subscriptions_data: str = "",
    ) -> str:
        """Store GraphQL endpoint with schema information as nodes and relationships.
        
        Args:
            url: GraphQL endpoint URL
            schema_info: Raw schema introspection data
            types_data: JSON string of GraphQL types
            queries_data: JSON string of available queries
            mutations_data: JSON string of available mutations
            subscriptions_data: JSON string of available subscriptions
            
        Returns:
            Confirmation message with endpoint ID
        """
        driver = await self._get_driver()
        
        try:
            # Create GraphQL endpoint node
            endpoint_query = """
            MERGE (e:GraphQLEndpoint {url: $url})
            SET e.timestamp = $timestamp,
                e.schema_info = $schema_info,
                e.endpoint_id = coalesce(e.endpoint_id, randomUUID())
            RETURN e.endpoint_id as endpoint_id
            """
            
            endpoint_params = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "schema_info": schema_info,
            }
            
            async with driver.session() as session:
                result = await session.run(endpoint_query, endpoint_params)
                endpoint_record = await result.single()
                endpoint_id = endpoint_record['endpoint_id'] if endpoint_record else None
                
                # Parse and store types if provided
                if types_data:
                    import json
                    try:
                        types = json.loads(types_data)
                        for type_info in types:
                            type_name = type_info.get('name', 'Unknown')
                            type_query = """
                            MATCH (e:GraphQLEndpoint {endpoint_id: $endpoint_id})
                            MERGE (t:GraphQLType {name: $type_name, endpoint_url: $url})
                            SET t.kind = $kind,
                                t.description = $description,
                                t.type_id = coalesce(t.type_id, randomUUID())
                            MERGE (e)-[:HAS_TYPE]->(t)
                            RETURN t.type_id as type_id
                            """
                            
                            type_params = {
                                "endpoint_id": endpoint_id,
                                "url": url,
                                "type_name": type_name,
                                "kind": type_info.get('kind', ''),
                                "description": type_info.get('description', ''),
                            }
                            
                            await session.run(type_query, type_params)
                            
                            # Store fields for this type
                            fields = type_info.get('fields', [])
                            for field in fields:
                                field_name = field.get('name', 'Unknown')
                                field_query = """
                                MATCH (t:GraphQLType {name: $type_name, endpoint_url: $url})
                                MERGE (f:GraphQLField {name: $field_name, type_name: $type_name, endpoint_url: $url})
                                SET f.field_type = $field_type,
                                    f.description = $field_description,
                                    f.is_deprecated = $is_deprecated,
                                    f.field_id = coalesce(f.field_id, randomUUID())
                                MERGE (t)-[:HAS_FIELD]->(f)
                                """
                                
                                field_params = {
                                    "url": url,
                                    "type_name": type_name,
                                    "field_name": field_name,
                                    "field_type": str(field.get('type', {})),
                                    "field_description": field.get('description', ''),
                                    "is_deprecated": field.get('isDeprecated', False),
                                }
                                
                                await session.run(field_query, field_params)
                    except json.JSONDecodeError:
                        pass
                
                # Store queries if provided
                if queries_data:
                    try:
                        queries = json.loads(queries_data)
                        for query in queries:
                            query_name = query.get('name', 'Unknown')
                            query_query = """
                            MATCH (e:GraphQLEndpoint {endpoint_id: $endpoint_id})
                            MERGE (q:GraphQLQuery {name: $query_name, endpoint_url: $url})
                            SET q.description = $description,
                                q.return_type = $return_type,
                                q.args = $args,
                                q.query_id = coalesce(q.query_id, randomUUID())
                            MERGE (e)-[:HAS_QUERY]->(q)
                            """
                            
                            query_params = {
                                "endpoint_id": endpoint_id,
                                "url": url,
                                "query_name": query_name,
                                "description": query.get('description', ''),
                                "return_type": str(query.get('type', {})),
                                "args": str(query.get('args', [])),
                            }
                            
                            await session.run(query_query, query_params)
                    except json.JSONDecodeError:
                        pass
                
                # Store mutations if provided
                if mutations_data:
                    try:
                        mutations = json.loads(mutations_data)
                        for mutation in mutations:
                            mutation_name = mutation.get('name', 'Unknown')
                            mutation_query = """
                            MATCH (e:GraphQLEndpoint {endpoint_id: $endpoint_id})
                            MERGE (m:GraphQLMutation {name: $mutation_name, endpoint_url: $url})
                            SET m.description = $description,
                                m.return_type = $return_type,
                                m.args = $args,
                                m.mutation_id = coalesce(m.mutation_id, randomUUID())
                            MERGE (e)-[:HAS_MUTATION]->(m)
                            """
                            
                            mutation_params = {
                                "endpoint_id": endpoint_id,
                                "url": url,
                                "mutation_name": mutation_name,
                                "description": mutation.get('description', ''),
                                "return_type": str(mutation.get('type', {})),
                                "args": str(mutation.get('args', [])),
                            }
                            
                            await session.run(mutation_query, mutation_params)
                    except json.JSONDecodeError:
                        pass
                
            logger.info(f"Stored GraphQL endpoint and schema for {url}")
            return f"Successfully stored GraphQL endpoint: {endpoint_id}"
            
        except Exception as e:
            logger.error(f"Failed to store GraphQL endpoint: {e}")
            return f"Failed to store GraphQL endpoint: {e}"

    @tool_method()
    async def store_graphql_finding(
        self,
        url: str,
        vulnerability_type: str,
        risk_level: str,
        schema_info: str,
        response_evidence: str,
        exposed_types: str = "",
        sensitive_fields: str = "",
        impact: str = "",
    ) -> str:
        """Store a GraphQL introspection vulnerability finding in Neo4j.
        
        Args:
            url: Target GraphQL endpoint URL
            vulnerability_type: Type of GraphQL vulnerability (e.g., "introspection_enabled", "schema_exposure")
            risk_level: Risk level (low/medium/high/critical)
            schema_info: Exposed schema information
            response_evidence: Raw response showing introspection data
            exposed_types: List of exposed GraphQL types
            sensitive_fields: Potentially sensitive field names
            impact: Potential impact description
            
        Returns:
            Confirmation message with finding ID
        """
        driver = await self._get_driver()
        
        try:
            query = """
            CREATE (f:GraphQLVulnerability {
                url: $url,
                vulnerability_type: $vulnerability_type,
                risk_level: $risk_level,
                timestamp: $timestamp,
                schema_info: $schema_info,
                response_evidence: $response_evidence,
                exposed_types: $exposed_types,
                sensitive_fields: $sensitive_fields,
                impact: $impact,
                finding_id: randomUUID()
            })
            RETURN f.finding_id as finding_id
            """
            
            params = {
                "url": url,
                "vulnerability_type": vulnerability_type,
                "risk_level": risk_level.upper(),
                "timestamp": datetime.now().isoformat(),
                "schema_info": schema_info,
                "response_evidence": response_evidence,
                "exposed_types": exposed_types,
                "sensitive_fields": sensitive_fields,
                "impact": impact,
            }
            
            async with driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
            logger.info(f"Stored GraphQL finding for {url}")
            return f"Successfully stored GraphQL finding: {record['finding_id'] if record else 'unknown'}"
            
        except Exception as e:
            logger.error(f"Failed to store GraphQL finding: {e}")
            return f"Failed to store GraphQL finding: {e}"
    
    @tool_method()
    async def query_findings(
        self,
        subdomain: str | None = None,
        risk_level: str | None = None,
        limit: int = 100,
    ) -> str:
        """Query stored vulnerability findings.
        
        Args:
            subdomain: Filter by specific subdomain
            risk_level: Filter by risk level (HIGH, MEDIUM, LOW)
            limit: Maximum number of results
            
        Returns:
            JSON string of findings
        """
        try:
            driver = await self._get_driver()
            
            where_clauses = []
            params = {"limit": limit}
            
            if subdomain:
                where_clauses.append("f.subdomain = $subdomain")
                params["subdomain"] = subdomain
                
            if risk_level:
                where_clauses.append("f.risk_level = $risk_level")
                params["risk_level"] = risk_level.upper()
            
            where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            query = f"""
            MATCH (s:Subdomain)-[:HAS_FINDING]->(f:Finding)
            {where_clause}
            RETURN f
            ORDER BY f.discovered_at DESC
            LIMIT $limit
            """
            
            async with driver.session() as session:
                result = await session.run(query, params)
                findings = [record["f"] async for record in result]
                
            logger.info(f"Retrieved {len(findings)} findings")
            return f"Found {len(findings)} findings: {findings}"
            
        except Exception as e:
            logger.error(f"Failed to query findings: {e}")
            return f"Failed to query findings: {e}"
    
    @tool_method()  
    async def run_cypher_query(self, query: str, params: dict[str, t.Any] | None = None) -> str:
        """Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            Query results as string
        """
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                result = await session.run(query, params or {})
                records = [record.data() async for record in result]
                
            logger.info(f"Executed query, returned {len(records)} records")
            return f"Query results: {records}"
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Query failed: {e}"
    
    async def close(self):
        """Close Neo4j driver connection."""
        if hasattr(self, '_driver') and self._driver:
            await self._driver.close()
            self._driver = None