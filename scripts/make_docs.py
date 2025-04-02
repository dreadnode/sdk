#!/usr/bin/env python3

import argparse
import builtins
import inspect
import io
import os
import pkgutil
import pydoc
import re
import sys
import types
import typing as t
from importlib import import_module
from pathlib import Path

# Import the docstring parser library
from docstring_parser import parse as parse_docstring
from docstring_parser import ParseError, DocstringStyle

# --- Helper Functions ---
def get_raw_doc(obj: t.Any) -> str | None:
    """Gets the raw, uncleaned docstring."""
    return inspect.getdoc(obj)

def format_signature(obj: t.Any, is_method: bool = False, class_name: str | None = None) -> str:
    """Formats the signature of a callable object for code display."""
    try:
        actual_obj = obj
        if isinstance(obj, property): actual_obj = obj.fget if obj.fget else obj
        elif isinstance(obj, (staticmethod, classmethod)): actual_obj = getattr(obj, '__func__', obj)
        if not callable(actual_obj): return ""
        sig = inspect.signature(actual_obj)
        sig_str = str(sig)
        if class_name:
            sig_str = re.sub(rf'\b{re.escape(class_name)}\.([\w]+)\b', r'\1', sig_str)
        return sig_str
    except (ValueError, TypeError): return "(...)"
    except Exception as e:
        print(f"Warning: Could not get signature for {getattr(obj, '__name__', 'unknown')}: {e}", file=sys.stderr)
        return "(...)"

def format_type_annotation(annotation_str: str) -> str:
    """Convert pipe-style union types to MDX-compatible format."""
    if '|' in annotation_str:
        # Handle simple Optional types
        if ' | None' in annotation_str:
            base_type = annotation_str.replace(' | None', '').strip()
            return f"Optional[{base_type}]"
        
        # General case for union types
        parts = [part.strip() for part in annotation_str.split('|')]
        return f"Union[{', '.join(parts)}]"
    
    return annotation_str

# --- Core MDX Generator Class ---

class MDXDoc(pydoc.HTMLDoc):
    """Formatter class for creating clean, readable MDX documentation."""

    # --- Docstring Formatting Logic ---
    def _format_docstring(self, obj: t.Any) -> str:
        """Parses and formats the docstring using a clean, traditional style with concise sections."""
        raw_doc = get_raw_doc(obj)
        if not raw_doc:
            return "*(No documentation provided)*\n\n"

        try:
            parsed = parse_docstring(raw_doc, style=DocstringStyle.GOOGLE)
            output = io.StringIO()

            # Description
            description = ""
            if parsed.short_description: description += parsed.short_description
            if parsed.long_description:
                if description: description += "\n\n"
                description += parsed.long_description
            if description:
                 output.write(description.replace('<', r'\<') + "\n\n")

            # Parameters - Simple list with inline details
            if parsed.params:
                output.write("**Parameters:**\n\n")
                
                for param in parsed.params:
                    # Parameter name and type
                    param_header = f"**`{param.arg_name}`**"
                    if param.type_name:
                        safe_type = param.type_name.replace('`', r'\`').replace('<', r'\<')
                        safe_type = format_type_annotation(safe_type)
                        param_header += f" (`{safe_type}`)"
                    if param.is_optional:
                        param_header += " *(optional)*"
                    output.write(f"- {param_header}")
                    
                    # Description on same line
                    if param.description:
                        output.write(f": {param.description.replace('<', r'\<')}")
                    
                    # Default value
                    if param.default:
                        safe_default = param.default.replace('`', r'\`')
                        output.write(f" Default: `{safe_default}`")
                    
                    output.write("\n")
                
                output.write("\n")

            # Returns - Inline with heading
            if parsed.returns:
                # More concise format for returns
                return_line = "**Returns:** "
                if parsed.returns.type_name:
                    safe_type = parsed.returns.type_name.replace('`', r'\`').replace('<', r'\<')
                    safe_type = format_type_annotation(safe_type)
                    return_line += f"`{safe_type}`"
                if parsed.returns.description:
                    if parsed.returns.type_name:
                        return_line += " — "
                    return_line += parsed.returns.description.replace('<', r'\<')
                output.write(return_line + "\n\n")

            # Raises - Inline format
            if parsed.raises:
                output.write("**Raises:**\n\n")
                for exc in parsed.raises:
                    exc_line = "- "
                    if exc.type_name:
                        safe_type = exc.type_name.replace('`', r'\`').replace('<', r'\<')
                        safe_type = format_type_annotation(safe_type)
                        exc_line += f"`{safe_type}`"
                    if exc.description:
                        if exc.type_name:
                            exc_line += " — "
                        exc_line += exc.description.replace('<', r'\<')
                    output.write(exc_line + "\n")
                output.write("\n")

            return output.getvalue()

        except ParseError as e:
            print(f"Warning: Could not parse docstring for {getattr(obj, '__name__', 'object')}: {e}", file=sys.stderr)
            return raw_doc.replace('<', r'\<') + "\n\n"
        except Exception as e:
            print(f"Error formatting docstring for {getattr(obj, '__name__', 'object')}: {e}", file=sys.stderr)
            return raw_doc.replace('<', r'\<') + "\n\n"

    # --- Overridden pydoc methods ---

    def page(self, title: str, contents: str) -> str:
        safe_title = title.replace("'", "''")
        return f"---\ntitle: '{safe_title}'\n---\n\n{contents}"

    def heading(self, title: str, level: int = 1) -> str:
        return f"{'#' * level} {title}\n"

    def section(self, title: str, contents: str, level: int = 2) -> str:
        return f"\n{'#' * level} {title}\n\n{contents}\n"

    def docmodule(self, object: types.ModuleType, name: str | None = None, mod: str | None = None, *ignored: t.Any) -> str:
        full_name = object.__name__
        short_name = full_name.split('.')[-1]
        safe_short_name = short_name.replace("'", "''")
        output = io.StringIO()

        output.write(f"---\ntitle: '{safe_short_name}'\nsidebarTitle: '{safe_short_name}'\n---\n\n")
        output.write(f"# Module `{short_name}`\n\n")
        output.write(f"*(Full name: `{full_name}`)*\n\n")

        try:
            source_file = inspect.getsourcefile(object)
            if source_file: 
                output.write(f"**Source file:** `{os.path.basename(source_file)}`\n\n")
        except (TypeError, OSError): 
            pass
        except Exception as e: 
            print(f"Warning: Could not get source file: {e}", file=sys.stderr)

        module_doc_formatted = self._format_docstring(object)
        output.write(module_doc_formatted)

        classes, functions = [], []
        try:
            for member_name, member_obj in inspect.getmembers(object):
                if member_name.startswith('_') and not member_name.startswith('__'): 
                    continue
                try: 
                    member_module = inspect.getmodule(member_obj)
                except Exception: 
                    member_module = None
                is_defined_here = member_module is not None and member_module.__name__ == full_name
                if is_defined_here:
                    if inspect.isclass(member_obj): 
                        classes.append((member_name, member_obj))
                    elif inspect.isfunction(member_obj): 
                        functions.append((member_name, member_obj))
        except Exception as e:
            output.write(f"\n**Warning:** Error inspecting members: {e}\n\n")

        if classes:
            output.write("## Classes\n\n")
            for class_name, class_obj in sorted(classes, key=lambda item: item[0]):
                output.write(self.docclass(class_obj, class_name, module_name=full_name))
            output.write("\n")
            
        if functions:
            output.write("## Functions\n\n")
            for func_name, func_obj in sorted(functions, key=lambda item: item[0]):
                output.write(self.docroutine(func_obj, func_name, is_method=False, class_name=None, module_name=full_name))
            output.write("\n")
            
        return output.getvalue()

    def docclass(self, object: type, name: str | None = None, module_name: str | None = None, *ignored: t.Any) -> str:
        real_name = name or object.__name__
        output = io.StringIO()
        output.write(f"\n### Class `{real_name}`\n\n")

        if object.__bases__:
            bases = []
            for b in object.__bases__:
                if b is object or (b is builtins.object and len(object.__bases__) > 1): 
                    continue
                base_module = getattr(b, '__module__', '')
                base_name_str = getattr(b, '__name__', str(b))
                if base_module and base_module != 'builtins': 
                    bases.append(f"`{base_module}.{base_name_str}`")
                else: 
                    bases.append(f"`{base_name_str}`")
            if bases:
                output.write(f"**Inherits from:** {', '.join(bases)}\n\n")

        class_doc_formatted = self._format_docstring(object)
        output.write(class_doc_formatted)

        methods, properties = [], []
        try:
            for member_name, member_obj in inspect.getmembers(object):
                if member_name.startswith('_') and not member_name.startswith('__'): 
                    continue
                is_directly_defined = member_name in object.__dict__
                member_origin_module = None
                try:
                    target_obj = member_obj
                    if isinstance(member_obj, property): 
                        target_obj = member_obj.fget or member_obj
                    elif isinstance(member_obj, (staticmethod, classmethod)): 
                        target_obj = getattr(member_obj, '__func__', None)
                    if target_obj: 
                        member_origin_module = inspect.getmodule(target_obj)
                except Exception: 
                    pass
                is_relevant = is_directly_defined or (member_origin_module and member_origin_module.__name__ == module_name)
                if not is_relevant: 
                    continue

                if isinstance(member_obj, property): 
                    properties.append((member_name, member_obj))
                else:
                    is_method_type, underlying_func, is_instance_method = False, None, False
                    if inspect.isfunction(member_obj): 
                        is_method_type, underlying_func, is_instance_method = True, member_obj, True
                    elif isinstance(member_obj, classmethod): 
                        is_method_type, underlying_func = True, getattr(member_obj, '__func__', None)
                    elif isinstance(member_obj, staticmethod): 
                        is_method_type, underlying_func = True, getattr(member_obj, '__func__', None)
                    if is_method_type: 
                        methods.append((member_name, underlying_func or member_obj, is_instance_method))
        except Exception as e:
            output.write(f"\n**Warning:** Error inspecting members of {real_name}: {e}\n\n")

        # Properties (Simple list)
        if properties:
            output.write("#### Properties\n\n")
            for prop_name, prop_obj in sorted(properties, key=lambda item: item[0]):
                output.write(self._docproperty(prop_obj, prop_name, class_name=real_name))
            
        # Methods (Simple list)
        if methods:
            output.write("#### Methods\n\n")
            for method_name, method_obj, is_instance_method in sorted(methods, key=lambda item: item[0]):
                output.write(self.docroutine(method_obj, method_name, is_method=is_instance_method, 
                                            class_name=real_name, module_name=module_name))
            
        return output.getvalue()

    def docroutine(self, object: t.Any, name: str | None = None, is_method: bool = False, 
                  class_name: str | None = None, module_name: str | None = None, *ignored: t.Any) -> str:
        real_name = name or getattr(object, '__name__', 'unknown_routine')
        output = io.StringIO()
        
        # Heading level based on context (class method vs standalone function)
        heading_level = 5 if class_name else 3
        output.write(f"{'#' * heading_level} `{real_name}`\n\n") 

        # Function/method signature
        signature = format_signature(object, is_method=is_method, class_name=class_name)
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
        if hasattr(target_for_type, '__annotations__'):
            try:
                return_annotation = t.get_type_hints(target_for_type).get('return')
                if return_annotation:
                    annotation = str(return_annotation)
                    annotation = re.sub(r'\btyping\.', '', annotation)
                    if class_name: 
                        annotation = re.sub(rf'\b{re.escape(class_name)}\.', '', annotation)
                    annotation = format_type_annotation(annotation)
                    safe_annotation = annotation.replace('`', r'\`').replace('<', r'\<')
                    type_hint_str = f"`{safe_annotation}`"
            except Exception: 
                pass
        
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
        if prop.fget: details.append("getter")
        if prop.fset: details.append("setter")
        if prop.fdel: details.append("deleter")
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
        return text.replace('<', r'\<')
    
    def preformat(self, text: str) -> str: 
        return f"```\n{text}\n```"
    
    def multicolumn(self, list_items: t.List[t.Any], fmt: t.Callable) -> str: 
        return "\n".join(f"- {fmt(item)}" for item in list_items)
    
    def grey(self, text: str) -> str: 
        return text
    
    def write(self, *args, **kwargs): 
        pass

# --- Main execution logic ---
def generate_mdx_docs(module_paths: t.List[str], output_dir: str, project_root: str | None = None):
    """Generates clean, traditional MDX documentation for Python modules."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.resolve()}")
    mdx_formatter = MDXDoc()
    _project_root_path = Path(project_root).resolve() if project_root else Path.cwd()
    if str(_project_root_path) not in sys.path: 
        sys.path.insert(0, str(_project_root_path))
    if os.getcwd() not in sys.path: 
        sys.path.insert(0, os.getcwd())
    processed_modules = set()
    
    for path_str in module_paths:
        path = Path(path_str).resolve()
        print(f"Processing path: {path}")
        if path.is_file() and path.suffix == '.py' and path.name != '__init__.py':
            module_name = None
            try:
                best_match_len = -1
                for p_str in sys.path:
                    p = Path(p_str).resolve()
                    try:
                        rel_path = path.relative_to(p)
                        if '..' not in rel_path.parts:
                             current_len = len(p.parts)
                             if current_len > best_match_len:
                                 best_match_len = current_len
                                 module_name_parts = list(rel_path.parts[:-1]) + [path.stem]
                                 module_name = '.'.join(part for part in module_name_parts if part)
                    except ValueError: 
                        continue
                if not module_name:
                    module_name = path.stem
                    if str(path.parent) not in sys.path: 
                        sys.path.insert(0, str(path.parent))
            except Exception as e:
                 print(f"Warning: Error determining module name for {path}: {e}", file=sys.stderr)
                 module_name = path.stem
                 if str(path.parent) not in sys.path: 
                     sys.path.insert(0, str(path.parent))

            if not module_name: 
                continue

            print(f"  Attempting to import module: {module_name}")
            try:
                module = import_module(module_name)
                if module.__name__ in processed_modules: 
                    continue
                print(f"  Generating MDX for module: {module.__name__}")
                mdx_content = mdx_formatter.docmodule(module)
                output_filename = f"{module.__name__.replace('.', '/')}.mdx"
                output_file = output_path / output_filename
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f: 
                    f.write(mdx_content)
                print(f"  -> Wrote {output_file}")
                processed_modules.add(module.__name__)
            except ImportError as e: 
                print(f"Error importing module '{module_name}': {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing module {module_name}: {e}", file=sys.stderr)
                import traceback; traceback.print_exc()

        elif path.is_dir():
             print(f"  Processing directory as package: {path.name}")
             package_name = path.name
             if str(path.parent) not in sys.path: 
                 sys.path.insert(0, str(path.parent))
             for importer, modname, ispkg in pkgutil.walk_packages([str(path)], prefix=f"{package_name}."):
                 if modname in processed_modules: 
                     continue
                 print(f"  Attempting to import package module: {modname}")
                 try:
                      module = import_module(modname)
                      print(f"  Generating MDX for module: {module.__name__}")
                      mdx_content = mdx_formatter.docmodule(module)
                      output_filename = f"{module.__name__.replace('.', '/')}.mdx"
                      output_file = output_path / output_filename
                      output_file.parent.mkdir(parents=True, exist_ok=True)
                      with open(output_file, 'w', encoding='utf-8') as f: 
                          f.write(mdx_content)
                      print(f"  -> Wrote {output_file}")
                      processed_modules.add(module.__name__)
                 except ImportError as e: 
                     print(f"Error importing package module {modname}: {e}", file=sys.stderr)
                 except Exception as e:
                      print(f"Error processing package module {modname}: {e}", file=sys.stderr)
                      import traceback; traceback.print_exc()
        else: 
            print(f"Warning: Path is not Python file/directory: {path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simple, clean MDX documentation for Python modules.")
    parser.add_argument("modules", nargs='+', help="Paths to Python files or package directories.")
    parser.add_argument("-o", "--output-dir", default="mintlify_docs", help="Directory to write MDX files (default: mintlify_docs).")
    parser.add_argument("-p", "--project-root", default=None, help="Optional path to the project root directory (assists with import resolution). Defaults to CWD.")

    args = parser.parse_args()
    generate_mdx_docs(args.modules, args.output_dir, args.project_root)
    print("MDX generation complete.")