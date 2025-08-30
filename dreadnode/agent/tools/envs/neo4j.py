import asyncio
import logging
import types
import typing as t
from contextlib import AsyncExitStack
from pathlib import Path

import rigging as rg
import typing_extensions as te
from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase

from dreadnode.agent.tools import Toolset, tool_method
from dreadnode.agent.tools.envs.docker import (
    ContainerConfig,
    ContainerContext,
    HealthCheckConfig,
    container,
)

# Reduce Neo4j logging noise
logging.getLogger("neo4j").setLevel(logging.ERROR)

Mode = t.Literal["container", "external"]


class Neo4jTool(Toolset):
    """
    A high-level client for interacting with a Neo4j database.
    (Docstrings for modes remain the same)
    """

    def __init__(
        self,
        username: str = "neo4j",
        password: str = "password",  # noqa: S107
        uri: str | None = None,
        image: str = "neo4j:latest",
        data_dir: Path | str = ".neo4j",
        container_config: ContainerConfig | None = None,
    ):
        """
        Create a Neo4jTool instance using either a managed container or an external database URI.

        If `uri` is provided, it connects to an existing Neo4j database, otherwise a new container is started.

        Args:
            username: Neo4j database username.
            password: Neo4j database password.
            uri: Optional URI to connect to an existing Neo4j database.
            image: Docker image to use for the Neo4j container.
            data_dir: Directory to store Neo4j data when running in container mode.
            container_config: Optional configuration for the Neo4j container.
        """

        self.mode: t.Literal["container", "external"] = "external"
        self.image: str = image
        self.uri: str | None = uri
        self.auth: tuple[str, str] = ("neo4j", password)
        self.data_dir = Path(data_dir).absolute()
        self.container_config: ContainerConfig | None = None

        self._driver: AsyncDriver | None = None
        self._container_context = t.cast(
            "t.AsyncContextManager[ContainerContext]", AsyncExitStack()
        )

        if self.uri is None:
            self.mode = "container"
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.container_config = ContainerConfig(
                ports=[7687, 7474],
                env={"NEO4J_AUTH": f"{username}/{password}"},
                hostname="neo4j",
                volumes={self.data_dir: "/data"},
                health_check=HealthCheckConfig(
                    command=[
                        "cypher-shell",
                        "-u",
                        "neo4j",
                        "-p",
                        password,
                        "-d",
                        "neo4j",
                        "RETURN 1",
                    ],
                    interval_seconds=1,
                    retries=15,
                    start_period_seconds=3,
                ),
            ).merge(container_config)

            self._container_context = container(self.image, config=self.container_config)

    # 1
    #   - Neo4j starts a container - driver connects to the container
    #   - BBOT runs in a container - BBOT connects to the other (neo4j) container (swap for host.docker.internal)
    #
    # 2
    #   - Neo4j URI is provided (probably from host context (localhost:7474)) - driver connects to the URI
    #   - BBOT runs in a container - BBOT connects to the host Neo4j instance (swap for host.docker.internal)
    #
    # 3
    #   - Neo4j starts in a container - driver connects to the container (always host context)
    #   - BBOT runs on the host - BBOT connects to the Neo4j container (don't swap host.docker.internal)
    #
    # 4
    #   - Neo4j URI is provided (probably from host context (localhost:7474)) - driver connects to the URI
    #   - BBOT runs on the host - BBOT connects to the host Neo4j instance (don't swap host.docker.internal)

    async def __aenter__(self) -> "te.Self":
        """
        Enters the context, starting a container or connecting to a URI.
        """
        if self.mode == "container":
            logger.info("Starting Neo4j container ...")
            ctx = await self._container_context.__aenter__()
            self.uri = f"bolt://localhost:{ctx.ports[7687]}"
            logger.success(f"Neo4j container started '{ctx.name}'")
            logger.info(f" |- Dashboard: http://localhost:{ctx.ports[7474]}")
            logger.info(f" |- Bolt URI:  {self.uri}")

        if not self.uri or not self.auth:
            raise RuntimeError("Internal state error: URI or auth not set.")

        self._driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)
        try:
            await self._driver.verify_connectivity()
            logger.success(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            await self._container_context.__aexit__(type(e), e, e.__traceback__)
            raise

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """
        Exits the context, closing the driver and cleaning up the container if managed.
        """
        if self._driver:
            await self._driver.close()
        await self._container_context.__aexit__(exc_type, exc_val, exc_tb)

    @rg.tool_method(catch=True)
    async def query(
        self, cypher: str, params: dict[str, t.Any] | None = None
    ) -> list[dict[str, t.Any]]:
        """
        Runs a Cypher query against the connected Neo4j instance.

        Args:
            cypher: The Cypher query string to execute.
            params: Optional parameters for the query.
        """
        if not self._driver:
            raise RuntimeError("Neo4jTool must be used within an 'async with' block.")

        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]

    @rg.tool_method(catch=True)
    async def get_nodes(
        self, label: str, filters: dict[str, t.Any] | None = None, limit: int | None = 100
    ) -> list[dict[str, t.Any]]:
        """
        Fetches nodes by label with optional property filtering.

        Args:
            label: The node label to query (e.g., 'Person', 'Product').
            filters: A dictionary of property-value pairs for exact matching.
            limit: The maximum number of nodes to return.

        Returns:
            A list of nodes, where each node is a dictionary of its properties.
        """
        # We use a WHERE clause for the label to allow parameterization,
        # which is safer than f-string formatting for the label itself.
        cypher = f"""
            MATCH (n)
            WHERE $label IN labels(n)
            {"AND " + " AND ".join(f"n.`{key}` = ${key}" for key in filters) if filters else ""}
            RETURN n
            {f"LIMIT {limit}" if limit is not None else ""}
        """

        # Combine parameters
        params = {"label": label}
        if filters:
            params.update(filters)

        result = await self.query(cypher, params)
        return [record["n"] for record in result]

    @rg.tool_method(catch=True)
    async def get_schema(self) -> dict[str, t.Any]:
        """
        Retrieves a comprehensive schema of the Neo4j database.

        Returns a dictionary containing node labels, relationship types,
        properties for nodes and relationships, and all constraints and indexes.
        This is essential for understanding the data model and constructing
        effective queries.

        Returns:
            A dictionary with keys: 'node_labels', 'relationship_types',
            'node_properties', 'relationship_properties', 'constraints', 'indexes'.
        """

        queries = {
            "node_labels": "CALL db.labels() YIELD label",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType",
            "node_properties": "CALL db.schema.nodeTypeProperties()",
            "relationship_properties": "CALL db.schema.relTypeProperties()",
            "constraints": "SHOW CONSTRAINTS",
            "indexes": "SHOW INDEXES",
        }

        # Query and unpack results
        results = await asyncio.gather(*(self.query(q) for q in queries.values()))
        (
            node_labels_res,
            rel_types_res,
            node_props_res,
            rel_props_res,
            constraints_res,
            indexes_res,
        ) = results

        # Process the results
        schema: dict[str, t.Any] = {
            "node_labels": sorted([r["label"] for r in node_labels_res]),
            "relationship_types": sorted([r["relationshipType"] for r in rel_types_res]),
            "node_properties": {},
            "relationship_properties": {},
            "constraints": [dict(r) for r in constraints_res],
            "indexes": [dict(r) for r in indexes_res],
        }

        # Structure node properties
        for record in node_props_res:
            label = record.get("nodeType", "").lstrip(":")
            if not label:
                continue
            if label not in schema["node_properties"]:
                schema["node_properties"][label] = []

            schema["node_properties"][label].append(
                {
                    "property": record.get("propertyName"),
                    "types": record.get("propertyTypes"),
                    "mandatory": record.get("mandatory"),
                }
            )

        # Structure relationship properties
        for record in rel_props_res:
            rel_type = record.get("relType", "").lstrip(":")
            if not rel_type:
                continue
            if rel_type not in schema["relationship_properties"]:
                schema["relationship_properties"][rel_type] = []

            schema["relationship_properties"][rel_type].append(
                {
                    "property": record.get("propertyName"),
                    "types": record.get("propertyTypes"),
                    "mandatory": record.get("mandatory"),
                }
            )

        return schema

    @tool_method(catch=True)
    async def explore_nodes(
        self, label: str | None = None, property_filter: str | None = None, limit: int = 100
    ) -> list[dict[str, t.Any]]:
        """
        Interactively explore nodes in the graph database.

        A flexible tool for discovering and examining nodes when you're not sure
        exactly what you're looking for. Supports filtering by type and properties.

        Args:
            label: Node type to filter by (e.g., 'DNS_NAME', 'FINDING', 'URL').
                Use get_schema() to see all available labels.
            property_filter: Simple property filter
                - 'property=value' for exact match
                - 'property CONTAINS value' for substring match
            limit: Maximum nodes to return (default: 100, max: 1000).

        Returns:
            List of node records with all their properties.
        """
        if limit < 1 or limit > 1000:  # noqa: PLR2004
            raise ValueError("Limit must be between 1 and 1000.")

        query_parts = [f"MATCH (node:{label})" if label else "MATCH (node)"]
        params: dict[str, t.Any] = {}

        if property_filter:
            if "=" in property_filter:
                prop, value = property_filter.split("=", 1)
                query_parts.append(f"WHERE node.{prop} = $value")
                params["value"] = value.strip()
            elif "CONTAINS" in property_filter:
                parts = property_filter.split("CONTAINS", 1)
                if len(parts) == 2:  # noqa: PLR2004
                    prop, value = parts
                    query_parts.append(f"WHERE node.{prop.strip()} CONTAINS $value")
                    params["value"] = value.strip()

        query_parts.append("RETURN node LIMIT $limit")
        query = " ".join(query_parts)

        return await self.query(query, {"limit": limit, **params})

    @rg.tool_method(catch=True)
    async def explore_relationships(
        self,
        source_label: str | None = None,
        relationship_type: str | None = None,
        target_label: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, t.Any]]:
        """
        Discover how different nodes are connected in the graph database.

        Args:
            source_label: Type of source node (e.g., 'DNS_NAME', 'IP_ADDRESS').
            relationship_type: Relationship type (e.g., 'RESOLVES_TO', 'HAS_PORT').
                            Use get_schema() to see all types.
            target_label: Type of target node.
            limit: Maximum relationships to return (default: 100).

        Returns:
            List of relationships with source node, relationship properties,
            and target node.
        """
        if limit < 1 or limit > 1000:  # noqa: PLR2004
            raise ValueError("Limit must be between 1 and 1000.")

        # Build the match pattern
        source = f"(source:{source_label})" if source_label else "(source)"
        rel = f"-[relationship:{relationship_type}]->" if relationship_type else "-[relationship]->"
        target = f"(target:{target_label})" if target_label else "(target)"
        query = f"MATCH {source}{rel}{target} RETURN source, relationship, target LIMIT $limit"

        return await self.query(query, {"limit": limit})
