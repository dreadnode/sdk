# type: ignore  # noqa: PGH003


import argparse
import builtins
import inspect
import io
import json
import logging
import pkgutil
import pydoc
import re
import sys
import types
import typing as t
from importlib import import_module
from pathlib import Path

from docstring_parser import Docstring, DocstringStyle, ParseError
from docstring_parser import parse as parse_docstring

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


# --- Helper Functions ---
def get_raw_doc(obj: t.Any) -> str | None:
    """Gets the raw, uncleaned docstring."""
    return inspect.getdoc(obj)


def format_signature(obj: t.Any, class_name: str | None = None) -> str:
    """Formats the signature of a callable object for code display."""
    try:
        actual_obj = obj
        if isinstance(obj, property):
            actual_obj = obj.fget if obj.fget else obj
        elif isinstance(obj, (staticmethod, classmethod)):
            actual_obj = getattr(obj, "__func__", obj)
        if not callable(actual_obj):
            return ""
        sig = inspect.signature(actual_obj)
        sig_str = str(sig)
        if class_name:
            sig_str = re.sub(rf"\b{re.escape(class_name)}\.([\w]+)\b", r"\1", sig_str)
    except (ValueError, TypeError):
        return "(...)"
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Warning: Could not get signature for %s: %s", getattr(obj, "__name__", "unknown"), e
        )
        return "(...)"

    return sig_str


def format_type_annotation(annotation_str: str) -> str:
    """Convert pipe-style union types to MDX-compatible format."""
    if "|" in annotation_str:
        # Handle simple Optional types
        if " | None" in annotation_str:
            base_type = annotation_str.replace(" | None", "").strip()
            return f"Optional[{base_type}]"

        # General case for union types
        parts = [part.strip() for part in annotation_str.split("|")]
        return f"Union[{', '.join(parts)}]"

    return annotation_str


# --- Core MDX Generator Class ---


class MDXDoc(pydoc.HTMLDoc):
    """Formatter class for creating clean, readable MDX documentation."""

    def __init__(self, auth_group: str | None = None):
        super().__init__()
        self.auth_group = auth_group

    # --- Docstring Formatting Logic ---
    def _format_docstring(self, obj: t.Any) -> str:
        """Parses and formats the docstring using a clean, traditional style with concise sections."""
        raw_doc = get_raw_doc(obj)
        if not raw_doc:
            return ""

        try:
            parsed = parse_docstring(raw_doc, style=DocstringStyle.GOOGLE)
            output = io.StringIO()

            # Process each section of the docstring
            self._write_description(parsed, output)
            self._write_parameters(parsed, output)
            self._write_returns(parsed, output)
            self._write_raises(parsed, output)

            return output.getvalue()

        except ParseError as e:
            logger.warning(
                "Warning: Could not parse docstring for %s: %s",
                getattr(obj, "__name__", "object"),
                e,
            )
            return raw_doc.replace("<", r"\<") + "\n\n"
        except Exception as e:  # noqa: BLE001
            logger.info(
                "Error formatting docstring for %s: %s", getattr(obj, "__name__", "object"), e
            )
            return raw_doc.replace("<", r"\<") + "\n\n"

    def _write_description(self, parsed: Docstring, output: io.StringIO) -> None:
        """Writes the description section of the docstring."""
        description = ""
        if parsed.short_description:
            description += parsed.short_description
        if parsed.long_description:
            if description:
                description += "\n\n"
            description += parsed.long_description
        if description:
            output.write(description.replace("<", r"\<") + "\n\n")

    def _write_parameters(self, parsed: Docstring, output: io.StringIO) -> None:
        """Writes the parameters section of the docstring."""
        if parsed.params:
            output.write("**Parameters:**\n\n")
            for param in parsed.params:
                param_header = f"**`{param.arg_name}`**"
                if param.type_name:
                    safe_type = param.type_name.replace("`", r"\`").replace("<", r"\<")
                    safe_type = format_type_annotation(safe_type)
                    param_header += f" (`{safe_type}`)"
                if param.is_optional:
                    param_header += " *(optional)*"
                output.write(f"- {param_header}")
                if param.description:
                    output.write(f": {param.description.replace('<', r'\\<')}")
                if param.default:
                    safe_default = param.default.replace("`", r"\`")
                    output.write(f" Default: `{safe_default}`")
                output.write("\n")
            output.write("\n")

    def _write_returns(self, parsed: Docstring, output: io.StringIO) -> None:
        """Writes the returns section of the docstring."""
        if parsed.returns:
            return_line = "**Returns:** "
            if parsed.returns.type_name:
                safe_type = parsed.returns.type_name.replace("`", r"\`").replace("<", r"\<")
                safe_type = format_type_annotation(safe_type)
                return_line += f"`{safe_type}`"
            if parsed.returns.description:
                if parsed.returns.type_name:
                    return_line += " — "
                return_line += parsed.returns.description.replace("<", r"\<")
            output.write(return_line + "\n\n")

    def _write_raises(self, parsed: Docstring, output: io.StringIO) -> None:
        """Writes the raises section of the docstring."""
        if parsed.raises:
            output.write("**Raises:**\n\n")
            for exc in parsed.raises:
                exc_line = "- "
                if exc.type_name:
                    safe_type = exc.type_name.replace("`", r"\`").replace("<", r"\<")
                    safe_type = format_type_annotation(safe_type)
                    exc_line += f"`{safe_type}`"
                if exc.description:
                    if exc.type_name:
                        exc_line += " — "
                    exc_line += exc.description.replace("<", r"\<")
                output.write(exc_line + "\n")
            output.write("\n")

    # --- Overridden pydoc methods ---

    def page(self, title: str, contents: str) -> str:
        safe_title = title.replace("'", "''")
        return f"---\ntitle: '{safe_title}'\n---\n\n{contents}"

    def heading(self, title: str, level: str = 1) -> str:
        return f"{'#' * level} {title}\n"

    def section(self, title: str, contents: str, level: int = 2) -> str:
        return f"\n{'#' * level} {title}\n\n{contents}\n"

    def docmodule(
        self,
        object: types.ModuleType,
    ) -> str:
        full_name = object.__name__
        short_name = full_name.split(".")[-1]
        safe_short_name = short_name.replace("'", "''")
        output = io.StringIO()

        # Write frontmatter and module header
        self._write_frontmatter(output, safe_short_name, short_name, full_name)

        # Write source file information
        self._write_source_file_info(output, object)

        # Write module docstring
        module_doc_formatted = self._format_docstring(object)
        output.write(module_doc_formatted)

        # Collect and document members
        classes, functions = self._collect_members(object, full_name)
        self._write_classes(output, classes, full_name)
        self._write_functions(output, functions)

        return output.getvalue()

    def _write_frontmatter(self, output, safe_short_name, short_name, full_name):
        """Writes the frontmatter and module header."""
        output.write(f"---\ntitle: '{safe_short_name}'\nsidebarTitle: '{safe_short_name}'\n")
        if self.auth_group:
            output.write(f'groups: ["{self.auth_group}"]\n')
        output.write("---\n\n")
        output.write(f"# Module `{short_name}`\n\n")
        output.write(f"*(Full name: `{full_name}`)*\n\n")

    def _write_source_file_info(self, output, object):
        """Writes the source file information."""
        try:
            source_file = inspect.getsourcefile(object)
            if source_file:
                output.write(f"**Source file:** `{Path(source_file).name}`\n\n")
        except (TypeError, OSError):
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("Warning: Could not get source file: %s", e)

    def _collect_members(self, object, full_name):
        """Collects classes and functions defined in the module."""
        classes, functions = [], []
        try:
            for member_name, member_obj in inspect.getmembers(object):
                if member_name.startswith("_") and not member_name.startswith("__"):
                    continue
                if self._is_defined_here(member_obj, full_name):
                    if inspect.isclass(member_obj):
                        classes.append((member_name, member_obj))
                    elif inspect.isfunction(member_obj):
                        functions.append((member_name, member_obj))
        except Exception as e:  # noqa: BLE001
            logger.warning("Error inspecting members: %s", e)
        return classes, functions

    def _is_defined_here(self, member_obj, full_name):
        """Checks if a member is defined in the current module."""
        try:
            member_module = inspect.getmodule(member_obj)
        except Exception:  # noqa: BLE001
            return False
        return member_module is not None and member_module.__name__ == full_name

    def _write_classes(self, output, classes, full_name):
        """Writes the documentation for classes."""
        if classes:
            output.write("## Classes\n\n")
            for class_name, class_obj in sorted(classes, key=lambda item: item[0]):
                output.write(self.docclass(class_obj, class_name, module_name=full_name))
            output.write("\n")

    def _write_functions(self, output, functions):
        """Writes the documentation for functions."""
        if functions:
            output.write("## Functions\n\n")
            for func_name, func_obj in sorted(functions, key=lambda item: item[0]):
                output.write(self.docroutine(func_obj, func_name, class_name=None))
            output.write("\n")

    def docclass(
        self, object: type, name: str | None = None, module_name: str | None = None
    ) -> str:
        real_name = name or object.__name__
        output = io.StringIO()
        output.write(f"\n### Class `{real_name}`\n\n")

        self._write_inheritance_info(object, output)
        self._write_class_docstring(object, output)

        methods, properties = self._collect_class_members(object, module_name)
        self._write_properties(properties, output, real_name)
        self._write_methods(methods, output, real_name)

        return output.getvalue()

    def _write_inheritance_info(self, object: type, output: io.StringIO):
        """Writes inheritance information for a class."""
        if object.__bases__:
            bases = []
            for b in object.__bases__:
                if b is object or (b is builtins.object and len(object.__bases__) > 1):
                    continue
                base_module = getattr(b, "__module__", "")
                base_name_str = getattr(b, "__name__", str(b))
                if base_module and base_module != "builtins":
                    bases.append(f"`{base_module}.{base_name_str}`")
                else:
                    bases.append(f"`{base_name_str}`")
            if bases:
                output.write(f"**Inherits from:** {', '.join(bases)}\n\n")

    def _write_class_docstring(self, object: type, output: io.StringIO):
        """Writes the formatted docstring for a class."""
        class_doc_formatted = self._format_docstring(object)
        output.write(class_doc_formatted)

    def _collect_class_members(self, object: type, module_name: str | None) -> tuple[list, list]:
        """Collects methods and properties of a class."""
        methods, properties = [], []
        try:
            for member_name, member_obj in inspect.getmembers(object):
                if member_name.startswith("_") and not member_name.startswith("__"):
                    continue
                if self._is_relevant_member(member_name, member_obj, object, module_name):
                    if isinstance(member_obj, property):
                        properties.append((member_name, member_obj))
                    elif self._is_method(member_obj):
                        methods.append((member_name, member_obj))
        except Exception as e:  # noqa: BLE001
            logger.warning("Error inspecting members of %s: %s", object.__name__, e)
        return methods, properties

    def _is_relevant_member(
        self, member_name: str, member_obj: t.Any, object: type, module_name: str | None
    ) -> bool:
        """Determines if a member is relevant for documentation."""
        is_directly_defined = member_name in object.__dict__
        try:
            target_obj = member_obj.fget if isinstance(member_obj, property) else member_obj
            member_origin_module = inspect.getmodule(target_obj)
        except Exception:  # noqa: BLE001
            return False

        return is_directly_defined or (
            member_origin_module and member_origin_module.__name__ == module_name
        )

    def _is_method(self, member_obj: t.Any) -> bool:
        """Checks if a member is a method."""
        return inspect.isfunction(member_obj) or isinstance(member_obj, (classmethod, staticmethod))

    def _write_properties(self, properties: list, output: io.StringIO, class_name: str):
        """Writes properties of a class."""
        if properties:
            output.write("#### Properties\n\n")
            for prop_name, prop_obj in sorted(properties, key=lambda item: item[0]):
                output.write(self._docproperty(prop_obj, prop_name, class_name=class_name))

    def _write_methods(self, methods: list, output: io.StringIO, class_name: str):
        """Writes methods of a class."""
        if methods:
            output.write("#### Methods\n\n")
            for method_name, method_obj in sorted(methods, key=lambda item: item[0]):
                output.write(self.docroutine(method_obj, method_name, class_name=class_name))

    def docroutine(
        self,
        object: t.Any,
        name: str | None = None,
        class_name: str | None = None,
    ) -> str:
        real_name = name or getattr(object, "__name__", "unknown_routine")
        output = io.StringIO()

        # Heading level based on context (class method vs standalone function)
        heading_level = 5 if class_name else 3
        output.write(f"{'#' * heading_level} `{real_name}`\n\n")

        # Function/method signature
        signature = format_signature(object, class_name=class_name)
        if signature and signature != "(...)":
            output.write(f"```python\n{real_name}{signature}\n```\n\n")
        elif real_name:
            output.write(f"`{real_name}(...)`\n\n")

        # Docstring content
        doc_formatted = self._format_docstring(object)
        output.write(doc_formatted)

        # Add a separator only if we're not at the end of a section
        if class_name:
            output.write("---\n\n")

        return output.getvalue()

    def _docproperty(self, prop: property, name: str, class_name: str | None = None) -> str:
        output = io.StringIO()
        output.write(f"##### `{name}`\n\n")

        # Get property type annotation
        type_hint_str = ""
        target_for_type = prop.fget if prop.fget else prop
        if hasattr(target_for_type, "__annotations__"):
            try:
                return_annotation = t.get_type_hints(target_for_type).get("return")
                if return_annotation:
                    annotation = str(return_annotation)
                    annotation = re.sub(r"\btyping\.", "", annotation)
                    if class_name:
                        annotation = re.sub(rf"\b{re.escape(class_name)}\.", "", annotation)
                    annotation = format_type_annotation(annotation)
                    safe_annotation = annotation.replace("`", r"\`").replace("<", r"\<")
                    type_hint_str = f"`{safe_annotation}`"
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Warning: Error getting type hint for property %s: %s",
                    getattr(prop, "__name__", "unknown_property"),
                    e,
                )

        # Show type compactly
        if type_hint_str:
            output.write(f"**Type:** {type_hint_str} *(property)*\n\n")
        else:
            output.write("*(property)*\n\n")

        # Documentation
        doc_obj = prop
        raw_doc = get_raw_doc(prop)
        if not raw_doc and prop.fget:
            doc_obj = prop.fget

        doc_formatted = self._format_docstring(doc_obj)
        output.write(doc_formatted)

        # Property details (compact)
        details = []
        if prop.fget:
            details.append("getter")
        if prop.fset:
            details.append("setter")
        if prop.fdel:
            details.append("deleter")
        if details:
            output.write(f"*Has: {', '.join(details)}*\n\n")

        # Add separator
        output.write("---\n\n")

        return output.getvalue()

    def link(self, text: str, url: str) -> str:
        return f"[{text}]({url})"

    def strong(self, text: str) -> str:
        return f"**{text}**"

    def emphasis(self, text: str) -> str:
        return f"*{text}*"

    def escape(self, text: str) -> str:
        return text.replace("<", r"\<")

    def preformat(self, text: str) -> str:
        return f"```\n{text}\n```"

    def multicolumn(self, list_items: list[t.Any], fmt: t.Callable) -> str:
        return "\n".join(f"- {fmt(item)}" for item in list_items)

    def grey(self, text: str) -> str:
        return text

    def write(self, *args, **kwargs):
        pass


# --- Main execution logic ---
def generate_mdx_docs(
    module_paths: list[str],
    output_dir: str,
    auth_group: str | None = None,
    project_root: str | None = None,
):
    """Generates clean, traditional MDX documentation for Python modules."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path.resolve())

    mdx_formatter = MDXDoc(auth_group=auth_group)
    _project_root_path = Path(project_root).resolve() if project_root else Path.cwd()
    _setup_sys_path(_project_root_path)

    processed_modules = set()
    generated_files = []

    for path_str in module_paths:
        path = Path(path_str).resolve()
        logger.info("Processing path: %s", path)
        if path.is_file() and path.suffix == ".py" and path.name != "__init__.py":
            _process_file(path, mdx_formatter, processed_modules, generated_files, output_path)
        elif path.is_dir():
            _process_directory(path, mdx_formatter, processed_modules, generated_files, output_path)
        else:
            logger.warning("Warning: Path is not Python file/directory: %s", path)

    _write_docs_json(generated_files, output_path)


def _setup_sys_path(project_root: Path):
    """Sets up the system path for module imports."""
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if Path.cwd() not in sys.path:
        sys.path.insert(0, Path.cwd())


def _process_file(path, mdx_formatter, processed_modules, generated_files, output_path):
    """Processes a single Python file."""
    module_name = _determine_module_name(path)
    if not module_name:
        return

    logger.info("  Attempting to import module: %s", module_name)
    try:
        module = import_module(module_name)
        if module.__name__ in processed_modules:
            return
        logger.info("  Generating MDX for module: %s", module.__name__)
        _generate_mdx(module, mdx_formatter, processed_modules, generated_files, output_path)
    except ImportError:
        logger.exception("Error importing module '%s'", module_name)
    except Exception:
        logger.exception("Error processing module %s", module_name)


def _process_directory(path, mdx_formatter, processed_modules, generated_files, output_path):
    """Processes a directory as a package."""
    logger.info("  Processing directory as package: %s", path.name)
    package_name = path.name
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    for _, modname, _ in pkgutil.walk_packages([str(path)], prefix=f"{package_name}."):
        if modname in processed_modules:
            continue
        logger.info("  Attempting to import package module: %s", modname)
        try:
            module = import_module(modname)
            logger.info("  Generating MDX for module: %s", module.__name__)
            _generate_mdx(module, mdx_formatter, processed_modules, generated_files, output_path)
        except ImportError:
            logger.exception("Error importing package module %s", modname)
        except Exception:
            logger.exception("Error processing package module %s", modname)


def _determine_module_name(path):
    """Determines the module name for a given file path."""
    try:
        best_match_len = -1
        module_name = None
        for p_str in sys.path:
            p = Path(p_str).resolve()
            try:
                rel_path = path.relative_to(p)
                if ".." not in rel_path.parts:
                    current_len = len(p.parts)
                    if current_len > best_match_len:
                        best_match_len = current_len
                        module_name_parts = [*list(rel_path.parts[:-1]), path.stem]
                        module_name = ".".join(part for part in module_name_parts if part)
            except ValueError:
                continue
        if not module_name:
            module_name = path.stem
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
        else:
            return module_name
    except Exception as e:  # noqa: BLE001
        logger.warning("Warning: Error determining module name for %s: %s", path, e)
        return None


def _generate_mdx(module, mdx_formatter, processed_modules, generated_files, output_path):
    """Generates MDX documentation for a module."""
    mdx_content = mdx_formatter.docmodule(module)
    output_filename = f"{module.__name__.replace('.', '/')}.mdx"
    output_file = output_path / output_filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(output_file, "w", encoding="utf-8") as f:
        f.write(mdx_content)
    logger.info("  -> Wrote %s", output_file)
    processed_modules.add(module.__name__)
    relative_path = str(output_file.relative_to(output_path.parent)).replace(".mdx", "")
    generated_files.append(relative_path)


def _write_docs_json(generated_files, output_path):
    """Writes the docs.json file."""
    docs_json_path = output_path / "docs.json"
    nav_file_paths = [
        str(file_path).replace(str(output_path) + "/", "") for file_path in generated_files
    ]
    for i, file_path in enumerate(nav_file_paths):
        if "dreadnode/" in file_path:
            parts = file_path.split("/")
            if len(parts) > 1 and parts[0] == "dreadnode":
                parts.insert(1, "library")
                nav_file_paths[i] = "/".join(parts)
    nested_pages = _build_nested_structure(nav_file_paths)
    docs_structure = {"group": "API Reference", "pages": nested_pages}
    with Path.open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(docs_structure, f, indent=2)
    logger.info("Generated navigation structure written to %s", docs_json_path)


def _build_nested_structure(file_paths, base_prefix="dreadnode/library"):
    """Builds a nested structure for the navigation based on file paths."""
    nested_structure = {}
    for file_path in file_paths:
        if file_path.startswith(base_prefix):
            relative_path = file_path[len(base_prefix) + 1 :]
            parts = relative_path.split("/")
            current_level = nested_structure
            for part in parts[:-1]:
                if part not in current_level:
                    current_level[part] = {}
                elif isinstance(current_level[part], str):
                    current_level[part] = {"index": current_level[part]}
                current_level = current_level[part]
            if parts[-1] in current_level and isinstance(current_level[parts[-1]], dict):
                if "dreadnode/" in file_path:
                    modified_path = file_path.replace("dreadnode/", "strikes/")
                    current_level[parts[-1]]["index"] = modified_path
                else:
                    current_level[parts[-1]]["index"] = file_path
            elif "dreadnode/" in file_path:
                modified_path = file_path.replace("dreadnode/", "strikes/")
                current_level[parts[-1]] = modified_path
            else:
                current_level[parts[-1]] = file_path

    def convert_to_list(structure):
        result = []
        for key, value in sorted(structure.items()):
            if isinstance(value, dict):
                result.append({"group": key, "pages": convert_to_list(value)})
            else:
                result.append(value)
        return result

    return convert_to_list(nested_structure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simple, clean MDX documentation for Python modules."
    )
    parser.add_argument("modules", nargs="+", help="Paths to Python files or package directories.")
    parser.add_argument(
        "-o", "--output-dir", default="docs", help="Directory to write MDX files (default: ./docs)."
    )
    parser.add_argument(
        "-p",
        "--project-root",
        default=None,
        help="Optional path to the project root directory (assists with import resolution). Defaults to CWD.",
    )
    parser.add_argument(
        "-g",
        "--auth-group",
        choices=["crucible", "strikes", "spyglass"],
        help="Optional authentication group to add to frontmatter.",
    )

    args = parser.parse_args()
    generate_mdx_docs(args.modules, args.output_dir, args.auth_group, args.project_root)
    logger.info("MDX generation complete.")
