#
# Fair warning, this file is a mess on the part of .NET interop. Order matters here for imports.
#

import asyncio
import functools
import hashlib
import sys
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

import rigging as rg
from loguru import logger
from pythonnet import load  # type: ignore [import-untyped]

import dreadnode as dn
from dreadnode.agent.tools import Toolset, tool_method

load("coreclr")

import clr  # type: ignore [import-untyped] # noqa: E402

lib_dir = Path(__file__).parent / "lib"
sys.path.append(str(lib_dir))

clr.AddReference("ICSharpCode.Decompiler")
clr.AddReference("Mono.Cecil")

# Helpers
# NEW imports (after Mono.Cecil):


from ICSharpCode.Decompiler import (  # type: ignore [import-not-found] # noqa: E402
    DecompilerSettings,
)
from ICSharpCode.Decompiler.CSharp import (  # type: ignore [import-not-found] # noqa: E402
    CSharpDecompiler,
)
from ICSharpCode.Decompiler.Metadata import (  # type: ignore [import-not-found] # noqa: E402
    MetadataTokenHelpers,
)
from Mono.Cecil import (  # type: ignore [import-not-found] # noqa: E402
    AssemblyDefinition,
    MethodReference,
)
from Mono.Cecil.Cil import Code  # type: ignore [import-not-found] # noqa: E402


# --- stable IDs so nodes dedupe nicely across runs ---
def _sid(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def _asm_id(asm_full_name: str) -> str:
    return f"assembly:{_sid(asm_full_name)}"


def _ns_id(asm_full_name: str, ns: str) -> str:
    return f"namespace:{_sid(asm_full_name, ns)}"


def _type_id(asm_full_name: str, type_full: str) -> str:
    return f"type:{_sid(asm_full_name, type_full)}"


def _meth_id(asm_full_name: str, type_full: str, meth_sig: str) -> str:
    return f"method:{_sid(asm_full_name, type_full, meth_sig)}"


def _ext_meth_id(meth_sig: str) -> str:
    return f"external_method:{_sid(meth_sig)}"


def _str_id(asm_full_name: str, s: str) -> str:
    return f"str:{_sid(asm_full_name, s)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shorten_dotnet_name(name: str) -> str:
    return name.split(" ")[-1].split("(")[0]


def _get_decompiler(path: Path | str) -> CSharpDecompiler:
    settings = DecompilerSettings()
    settings.ThrowOnAssemblyResolveErrors = False
    return CSharpDecompiler(str(path), settings)


def _decompile_token(path: Path | str, token: int) -> str:
    entity_handle = MetadataTokenHelpers.TryAsEntityHandle(token.ToUInt32())  # type: ignore [attr-defined]
    return _get_decompiler(path).DecompileAsString(entity_handle)  # type: ignore [no-any-return]


def _find_references(assembly: AssemblyDefinition, search: str) -> list[str]:
    flexible_search_strings = [
        search.lower(),
        search.lower().replace(".", "::"),
        search.lower().replace("::", "."),
    ]

    using_methods: set[str] = set()
    for module in assembly.Modules:
        methods = []
        for module_type in module.Types:
            for method in module_type.Methods:
                methods.append(method)

        for method in methods:
            if not method.HasBody:
                continue

            for instruction in method.Body.Instructions:
                intruction_str = str(instruction.Operand).lower()

                for _search in flexible_search_strings:
                    if _search in intruction_str:
                        using_methods.add(method.FullName)

    return list(using_methods)


def _extract_unique_call_paths(
    tree: dict[str, t.Any],
    current_path: list[str] | None = None,
) -> list[list[str]]:
    if current_path is None:
        current_path = []

    if not tree:  # Leaf node
        return [current_path] if current_path else []

    paths = []
    for method, subtree in tree.items():
        new_path = [method, *current_path]
        paths.extend(_extract_unique_call_paths(subtree, new_path))

    return paths


def build_call_graph_payload(
    self, path: t.Annotated[str, "The binary file path"]
) -> dict[str, t.Any]:
    """
    Produce a Graph JSON payload: {"nodes":[...], "edges":[...]} with labels:
        Assembly, Namespace, Type, Method, ExternalMethod, StringLiteral
    and edges: Contains, Calls, References, Uses (all as 'open' events at now()).
    """
    logger.info(f"build_call_graph_payload({path})")
    path = self._resolve_path(path)
    assembly = AssemblyDefinition.ReadAssembly(path)
    asm_full = str(assembly.Name.FullName)
    asm_uuid = _asm_id(asm_full)

    nodes: list[dict[str, t.Any]] = []
    edges: list[dict[str, t.Any]] = []
    seen_nodes: set[str] = set()

    def add_node(uuid: str, label: str, name: str, attributes: dict[str, t.Any] | None = None):
        if uuid in seen_nodes:
            return
        nodes.append({"uuid": uuid, "label": label, "name": name, "attributes": attributes or {}})
        seen_nodes.add(uuid)

    def add_edge(src: str, dst: str, etype: str, attrs: dict[str, t.Any] | None = None):
        edges.append(
            {
                "src_uuid": src,
                "dst_uuid": dst,
                "type": etype,
                "event_kind": "open",
                "event_ts": _now_iso(),
                "attributes": attrs or {},
            }
        )

    # Assembly node
    add_node(asm_uuid, "Assembly", asm_full, {"path": str(self.base_path / path)})

    # Iterate types/methods and IL
    for module in assembly.Modules:
        for tdef in module.Types:
            t_full = str(tdef.FullName)
            ns = ".".join(t_full.split(".")[:-1]) if "." in t_full else ""
            t_uuid = _type_id(asm_full, t_full)
            if ns:
                ns_uuid = _ns_id(asm_full, ns)
                add_node(ns_uuid, "Namespace", ns, {})
                add_edge(asm_uuid, ns_uuid, "Contains")
                add_node(
                    t_uuid,
                    "Type",
                    t_full,
                    {"isValueType": bool(getattr(tdef, "IsValueType", False))},
                )
                add_edge(ns_uuid, t_uuid, "Contains")
            else:
                add_node(
                    t_uuid,
                    "Type",
                    t_full,
                    {"isValueType": bool(getattr(tdef, "IsValueType", False))},
                )
                add_edge(asm_uuid, t_uuid, "Contains")

            for mdef in tdef.Methods:
                m_sig = str(mdef.FullName)  # includes signature
                m_uuid = _meth_id(asm_full, t_full, m_sig)
                add_node(m_uuid, "Method", m_sig, {"declaringType": t_full})
                add_edge(t_uuid, m_uuid, "Contains")

                if not getattr(mdef, "HasBody", False) or mdef.Body is None:
                    continue

                for instr in mdef.Body.Instructions:
                    try:
                        code = instr.OpCode.Code
                    except Exception:
                        logger.warning(
                            f"Failed to get OpCode for instruction in {mdef.FullName}: {instr}"
                        )
                        continue

                    # Calls
                    if code in (Code.Call, Code.Callvirt, Code.Newobj):
                        target = instr.Operand
                        if isinstance(target, MethodReference):
                            tgt_sig = str(target.FullName)
                            # internal vs external
                            tgt_uuid: str
                            try:
                                resolved = target.Resolve()
                                if resolved is not None and resolved.Module == module:
                                    tgt_decl_t = str(resolved.DeclaringType.FullName)
                                    tgt_uuid = _meth_id(asm_full, tgt_decl_t, tgt_sig)
                                    add_node(
                                        tgt_uuid,
                                        "Method",
                                        tgt_sig,
                                        {"declaringType": tgt_decl_t},
                                    )
                                else:
                                    tgt_uuid = _ext_meth_id(tgt_sig)
                                    add_node(tgt_uuid, "ExternalMethod", tgt_sig, {})
                            except Exception:
                                tgt_uuid = _ext_meth_id(tgt_sig)
                                add_node(tgt_uuid, "ExternalMethod", tgt_sig, {})
                            add_edge(m_uuid, tgt_uuid, "Calls")

                    # String literals
                    if code == Code.Ldstr:
                        lit = str(instr.Operand)
                        s_uuid = _str_id(asm_full, lit)
                        add_node(s_uuid, "StringLiteral", lit[:120], {})
                        add_edge(m_uuid, s_uuid, "Uses")

                    # Type references (rough heuristic via ldtoken)
                    if code == Code.Ldtoken:
                        ref = instr.Operand
                        try:
                            ref_name = str(ref.FullName)
                            ref_uuid = _type_id(asm_full, ref_name)
                            add_node(ref_uuid, "Type", ref_name, {"refOnly": True})
                            add_edge(m_uuid, ref_uuid, "References")
                        except Exception:
                            logger.warning(
                                f"Failed to resolve ldtoken operand in {mdef.FullName}: {instr}"
                            )

    return {"nodes": nodes, "edges": edges}

    # --- NEW: ingest call graph into temporal memory (pandas store) ---


def ingest_call_graph_to_memory(
    self,
    path: t.Annotated[str, "The binary file path"],
    group_id: t.Annotated[str, "Target memory group_id, e.g., 'code:myapp'"] = "code:default",
    episode_name: t.Annotated[str, "Name for this ingestion event"] = "dotnet-callgraph",
) -> dict[str, t.Any]:
    """
    Build payload and push it to the memory service as a JSON 'episode'.
    Produces 'open' events at now(); re-run later to record evolution.
    """
    logger.info(f"ingest_call_graph_to_memory({path}, group_id={group_id})")
    payload = self.build_call_graph_payload(path)

    # defer import to avoid import-order surprises
    # IMPORTANT: add_memory expects episode_body to be a JSON string
    import json

    from tgraph.tool_api import add_memory  # thin wrapper to MemoryService

    episode_body = json.dumps(payload)

    # Run the coroutine in a blocking context (your rg/dn wrapper already does to_thread)
    # but add_memory is async; use asyncio.run if you're outside an event loop,
    # here we keep it simple and just run it synchronously via asyncio.get_event_loop().
    async def _run():
        return await add_memory(
            name=episode_name,
            episode_body=episode_body,
            group_id=group_id,
            source="json",
            source_description="dotnet call graph (Mono.Cecil)",
        )

    # Execute in current thread (your wrapper wraps this in to_thread already)
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        resp = loop.run_until_complete(_run())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    return {
        "message": resp.get("message", "ok") if isinstance(resp, dict) else "ok",
        "nodes": len(payload["nodes"]),
        "edges": len(payload["edges"]),
        "group_id": group_id,
    }


# Tools

DEFAULT_EXCLUDE = [
    "mscorlib.dll",
]


@dataclass
class DotnetReversing(Toolset):
    base_path: Path
    binaries: list[str]

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        pattern: str = "**/*",
        exclude: list[str] = DEFAULT_EXCLUDE,
    ) -> "DotnetReversing":
        base_path = Path(path)
        if not base_path.exists():
            raise ValueError(f"Base path does not exist: {base_path}")

        binaries: list[str] = []
        for file_path in base_path.rglob(pattern):
            rel_path = file_path.relative_to(base_path)
            if not any(ex in str(rel_path) for ex in exclude):
                binaries.append(str(rel_path))

        if not binaries:
            raise ValueError(
                f"No binaries found in {base_path} ({pattern})",
            )

        return cls(base_path=base_path, binaries=binaries)

    @cached_property
    def tools(self) -> list[t.Callable[..., t.Any]]:
        def wrap(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
            @rg.tool(catch=True, truncate=10_000)
            @dn.task()
            @functools.wraps(func)
            async def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
                # Use asyncio.to_thread to run the function in a separate thread
                # and avoid blocking the event loop.
                return await asyncio.to_thread(func, *args, **kwargs)

            return wrapper

        return [
            wrap(func)
            for func in (
                self.decompile_module,
                self.decompile_type,
                self.decompile_methods,
                self.list_namespaces,
                self.list_types_in_namespace,
                self.list_methods_in_type,
                self.list_types,
                self.list_methods,
                self.search_for_references,
                self.get_call_flows_to_method,
            )
        ]

    def _resolve_path(self, path: str) -> str:
        rel_path = Path(path)
        full_path = self.base_path / path

        # If we are already in the base path, make it relative
        # before we check anything (can occur with repeated calls)
        if rel_path.is_relative_to(self.base_path):
            rel_path = rel_path.relative_to(self.base_path)

        if str(rel_path) not in self.binaries or not full_path.exists():
            raise ValueError(f"{path} is not available.")

        return str(full_path)

    @tool_method
    def decompile_module(self, path: t.Annotated[str, "The binary file path"]) -> str:
        """
        Decompile the entire module and return the decompiled code as a string.
        """
        logger.info(f"decompile_module({path})")
        path = self._resolve_path(path)
        return _get_decompiler(path).DecompileWholeModuleAsString()  # type: ignore [no-any-return]

    @tool_method
    def decompile_type(
        self,
        path: t.Annotated[str, "The binary file path"],
        type_name: t.Annotated[str, "The specific type to decompile"],
    ) -> str:
        """
        Decompile a specific type and return the decompiled code as a string.
        """
        logger.info(f"decompile_type({path}, {type_name})")
        path = self._resolve_path(path)
        return _get_decompiler(path).DecompileTypeAsString(type_name)  # type: ignore [no-any-return]

    @tool_method
    def decompile_methods(
        self,
        path: t.Annotated[str, "The binary file path"],
        method_names: t.Annotated[list[str], "List of methods to decompile"],
    ) -> dict[str, str]:
        """
        Decompile specific methods and return a dictionary with method names as keys and decompiled code as values.
        """
        logger.info(f"decompile_methods({path}, {method_names})")
        flexible_method_names = [_shorten_dotnet_name(name).lower() for name in method_names]
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        methods: dict[str, str] = {}
        for module in assembly.Modules:
            for module_type in module.Types:
                for method in module_type.Methods:
                    method_name = _shorten_dotnet_name(method.FullName).lower()
                    if method_name in flexible_method_names:
                        methods[method.FullName] = _decompile_token(path, method.MetadataToken)
        return methods

    @tool_method
    def list_namespaces(self, path: t.Annotated[str, "The binary file path"]) -> list[str]:
        """
        List all namespaces in the assembly.
        """
        logger.info(f"list_namespaces({path})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)

        namespaces = set()
        for module in assembly.Modules:
            for module_type in module.Types:
                if "." in module_type.FullName:
                    # Get namespace part (everything before the last dot)
                    namespace = ".".join(module_type.FullName.split(".")[:-1])
                    namespaces.add(namespace)
                else:
                    # Handle types without namespace (add as root)
                    namespaces.add("<root>")

        return sorted(namespaces)

    @tool_method
    def list_types_in_namespace(
        self,
        path: t.Annotated[str, "The binary file path"],
        namespace: t.Annotated[str, "The namespace to list types from"],
    ) -> list[str]:
        """
        List all types in the specified namespace.
        """
        logger.info(f"list_types_in_namespace({path}, {namespace})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)

        types = []
        for module in assembly.Modules:
            for module_type in module.Types:
                if namespace == "<root>":
                    # Handle types without namespace
                    if "." not in module_type.FullName or (
                        module_type.FullName.count(".") == 1
                        and module_type.FullName.endswith("Module")
                    ):
                        types.append(module_type.FullName)
                elif module_type.FullName.startswith(f"{namespace}."):
                    # Check if the type belongs directly to this namespace (not a sub-namespace)
                    remainder = module_type.FullName[len(namespace) + 1 :]
                    if "." not in remainder:
                        types.append(module_type.FullName)

        return types

    @tool_method
    def list_methods_in_type(
        self,
        path: t.Annotated[str, "The binary file path"],
        type_name: t.Annotated[str, "The full type name"],
    ) -> list[str]:
        """
        List all methods in the specified type.
        """
        logger.info(f"list_methods_in_type({path}, {type_name})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)

        methods = []
        for module in assembly.Modules:
            for module_type in module.Types:
                if module_type.FullName == type_name:
                    methods.extend([method.Name for method in module_type.Methods])
                    break

        return methods

    @tool_method
    def list_types(self, path: t.Annotated[str, "The binary file path"]) -> list[str]:
        """
        List all types in the assembly and return their full names.
        """
        logger.info(f"list_types({path})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        return [module_type.FullName for module in assembly.Modules for module_type in module.Types]

    @tool_method
    def list_methods(self, path: t.Annotated[str, "The binary file path"]) -> list[str]:
        """
        List all methods in the assembly and return their full names.
        """
        logger.info(f"list_methods({path})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        methods: list[str] = []
        for module in assembly.Modules:
            for module_type in module.Types:
                methods.extend([method.FullName for method in module_type.Methods])
        return methods

    @tool_method
    def search_for_references(
        self,
        path: t.Annotated[str, "The binary file path"],
        search: t.Annotated[str, "A flexible search string used to check called function names"],
    ) -> list[str]:
        """
        Locate all methods inside the assembly that reference the search string.

        This can be used to locate uses of a specific function or method anywhere in the assembly.
        """
        logger.info(f"search_for_references({path}, {search})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        return _find_references(assembly, search)

    @tool_method
    def search_by_name(
        self,
        path: t.Annotated[str, "The binary file path"],
        search: t.Annotated[str, "Search string to match against types and methods"],
    ) -> dict[str, list[str]]:
        """
        Search for types and methods in the assembly that match the search string.
        This can be used to locate types and methods by name.
        """
        logger.info(f"search_by_name({path}, {search})")

        results: dict[str, list[str]] = {
            "types": [],
            "methods": [],
        }

        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)

        search_lower = search.lower()

        # Type search
        for module in assembly.Modules:
            for module_type in module.Types:
                if search_lower in module_type.FullName.lower():
                    results["types"].append(module_type.FullName)

        # Method search
        for module in assembly.Modules:
            for module_type in module.Types:
                for method in module_type.Methods:
                    if search_lower in method.FullName.lower():
                        results["methods"].append(method.FullName)

        return results

    @tool_method
    def get_call_flows_to_method(
        self,
        paths: t.Annotated[
            list[str],
            "Paths of all .NET assemblies to consider as part of the search",
        ],
        method_name: t.Annotated[str, "Target method name"],
        *,
        max_depth: int = 10,
    ) -> list[list[str]]:
        """
        Find all unique call flows to the target method inside provided assemblies and
        return a nested list of method names representing the call paths.
        """
        logger.info(f"get_call_flows_to_method({paths}, {method_name})")
        assemblies = [AssemblyDefinition.ReadAssembly(self._resolve_path(path)) for path in paths]
        short_target_name = _shorten_dotnet_name(method_name)

        def build_tree(
            method_name: str,
            current_depth: int = 0,
            visited: set[str] | None = None,
        ) -> dict[str, t.Any]:
            visited = visited or set()
            if method_name in visited or current_depth > max_depth:
                return {}

            visited.add(method_name)
            tree = {}

            for assembly in assemblies:
                for caller in _find_references(assembly, method_name):
                    if caller not in visited:
                        tree[caller] = build_tree(
                            _shorten_dotnet_name(caller),
                            current_depth + 1,
                            visited.copy(),
                        )

            return tree

        call_tree = build_tree(short_target_name)
        return _extract_unique_call_paths(call_tree)
