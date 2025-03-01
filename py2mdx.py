import sys
import subprocess
import re
from pathlib import Path

def get_pdoc_output(module_name: str) -> str:
    """Get documentation for a single module."""
    try:
        result = subprocess.run(
            ["pdoc", module_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error processing {module_name}: {e.stderr}")
        return ""

def is_module_name(name: str) -> bool:
    """Check if the name looks like a module rather than a class."""
    # Check if the name contains uppercase letters after a dot
    parts = name.split('.')
    # Only consider it a module if all parts start with lowercase
    return all(part[0].islower() for part in parts if part)

def find_submodules(content: str, base_module: str) -> set:
    """Extract actual submodule names from content."""
    submodules = set()
    
    # Look for submodule section
    in_submodule_section = False
    for line in content.split('\n'):
        if 'Sub-modules' in line:
            in_submodule_section = True
            continue
        if in_submodule_section:
            if line.strip() and not line.startswith('----'):
                if '* ' in line:
                    module = line.strip('* ').strip()
                    if module.startswith(base_module) and is_module_name(module):
                        submodules.add(module)
            elif line.strip() == '':
                in_submodule_section = False
    
    return submodules

def convert_module_to_mdx(module_name: str, output_dir: Path, processed_modules: set):
    """Convert a module and its submodules to MDX."""
    if module_name in processed_modules:
        return
    
    print(f"Processing {module_name}...")
    processed_modules.add(module_name)
    
    # Get the main module documentation
    content = get_pdoc_output(module_name)
    if not content:
        return
    
    # Write main module MDX
    module_file = output_dir / f"{module_name.replace('.', '/')}.mdx"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    
    mdx_content = f"""---
title: "{module_name}"
description: "API documentation for {module_name}"
---

{content}"""
    
    module_file.write_text(mdx_content)
    print(f"Created {module_file}")
    
    # Process submodules
    submodules = find_submodules(content, module_name)
    for submodule in submodules:
        convert_module_to_mdx(submodule, output_dir, processed_modules)

def create_index(output_dir: Path, root_module: str):
    """Create an index.mdx file with links to all modules."""
    index_content = f"""---
title: "{root_module} Documentation"
description: "API documentation for {root_module}"
---

# {root_module} Documentation

## Modules

"""
    
    # Add links to all generated files
    for mdx_file in sorted(output_dir.glob("**/*.mdx")):
        if mdx_file.name != "index.mdx":
            rel_path = mdx_file.relative_to(output_dir)
            module_path = str(rel_path).replace("/", ".").replace(".mdx", "")
            index_content += f"- [{module_path}](./{rel_path})\n"
    
    index_file = output_dir / "index.mdx"
    index_file.write_text(index_content)
    print(f"Created {index_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdoc_mdx.py <module_name>")
        sys.exit(1)
    
    module_name = sys.argv[1]
    output_dir = Path("docs/mdx")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_modules = set()
    convert_module_to_mdx(module_name, output_dir, processed_modules)
    create_index(output_dir, module_name)

if __name__ == "__main__":
    main()