import os
from pathlib import Path
from jinja2 import Template
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel

class DocAttribute(BaseModel):
    name: str
    args: str

class DocTypeParam(BaseModel):
    name: str
    bounds: List[str]
    description: str

class DocField(BaseModel):
    name: str
    type: str
    description: str
    visibility: str
    attributes: Optional[List[DocAttribute]] = None

class DocParameter(BaseModel):
    name: str
    type: str
    description: str

class DocMethod(BaseModel):
    name: str
    description: str
    parameters: List[DocParameter]
    return_type: str
    return_description: str
    visibility: str
    is_async: bool
    is_static: bool
    attributes: Optional[List[DocAttribute]] = None
    type_params: Optional[List[DocTypeParam]] = None
    examples: Optional[List[str]] = None

class DocVariant(BaseModel):
    name: str
    fields: Optional[List[DocField]] = None
    description: str

class DocEnum(BaseModel):
    name: str
    description: str
    variants: List[DocVariant]
    methods: Optional[List[DocMethod]] = None
    visibility: str
    attributes: Optional[List[DocAttribute]] = None
    type_params: Optional[List[DocTypeParam]] = None

class DocStruct(BaseModel):
    name: str
    description: str
    fields: List[DocField]
    methods: Optional[List[DocMethod]] = None
    visibility: str
    attributes: Optional[List[DocAttribute]] = None
    type_params: Optional[List[DocTypeParam]] = None
    examples: Optional[List[str]] = None

class DocModule(BaseModel):
    """Represents a Rust module."""
    name: str
    description: str

    # Children of a module:
    structs: Optional[List[DocStruct]] = None
    enums: Optional[List[DocEnum]] = None
    methods: Optional[List[DocMethod]] = None
    submodules: Optional[List['DocModule']] = None

    # Module-level attributes, if needed:
    attributes: Optional[List[DocAttribute]] = None
    examples: Optional[List[str]] = None

class DocCrate(BaseModel):
    """Represents the top-level crate."""
    name: str
    description: str
    # A crate can have multiple top-level modules:
    modules: List[DocModule]
    examples: Optional[List[str]] = None


def read_rust_files() -> List[dict[str, str]]:
    """Read all tracked Rust source files in the repository."""
    contents = []
    
    # Use git ls-files to get all tracked .rs files
    import subprocess
    result = subprocess.run(['git', 'ls-files', '*.rs'], 
                          capture_output=True, text=True, check=True)
    
    # Process each file path
    for file_path in sorted(result.stdout.splitlines()):
        with open(file_path, 'r', encoding='utf-8') as f:
            contents.append({
                'path': file_path,
                'content': f.read()
            })
    
    return contents

def convert_documentation(client: OpenAI, prompt: str, documentation: str) -> DocCrate:
    """Convert documentation to structured format using OpenAI."""
    print('Converting documentation...')
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": documentation}
        ],
        response_format=DocCrate
    )
    
    # Parse JSON response into DocCrate model
    return response.choices[-1].message.parsed

def main():
    # Read all Rust source files
    rust_files = read_rust_files()
    
    # Combine all Rust content
    all_docs = '\n\n'.join(f"# File: {f['path']}\n{f['content']}" 
                          for f in rust_files)
    
    # Read the template
    with open('.github/workflows/prompt.jinja', 'r') as f:
        template = Template(f.read())
    
    # Render the template
    prompt = template.render()
    
    # Create docs directory
    os.makedirs('markdown_docs', exist_ok=True)
    
    # Convert to structured documentation using OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    try:
        doc_crate = convert_documentation(client, prompt, all_docs)
        
        # Save both JSON and markdown versions
        with open('markdown_docs/documentation.json', 'w') as f:
            f.write(doc_crate.model_dump_json(indent=2))
            
        # Generate markdown from structured data
        markdown_content = generate_markdown(doc_crate)
        with open('markdown_docs/documentation.md', 'w') as f:
            f.write(markdown_content)
            
        print('Created documentation in both structured and markdown formats')
        
    except Exception as e:
        print(f'Error converting documentation: {str(e)}')
        raise

def format_visibility(visibility: str) -> str:
    """Format visibility modifier."""
    return f"pub" if visibility == "public" else "pub(crate)" if visibility == "crate" else "priv"

def format_type_params(type_params: Optional[List[DocTypeParam]]) -> str:
    """Format type parameters with bounds."""
    if not type_params:
        return ""
    params = []
    for p in type_params:
        if p.bounds:
            params.append(f"{p.name}: {' + '.join(p.bounds)}")
        else:
            params.append(p.name)
    return f"<{', '.join(params)}>"

def format_method_header(method: DocMethod, level: int = 3) -> str:
    """Format method name for header."""
    return f"{'#' * level} {method.name}"

def format_method_signature(method: DocMethod) -> str:
    """Format method signature as Rust code."""
    parts = []
    
    # Add modifiers
    modifiers = []
    if method.is_static:
        modifiers.append("static")
    if method.is_async:
        modifiers.append("async")
    if modifiers:
        parts.append(" ".join(modifiers))
    
    # Add visibility and fn
    parts.append(format_visibility(method.visibility))
    parts.append("fn")
    parts.append(method.name)
    
    # Add type parameters if present
    type_params = format_type_params(method.type_params)
    if type_params:
        parts.append(type_params)
    
    # Format parameters with proper spacing
    params = []
    for param in method.parameters:
        params.append(f"{param.name}: {param.type}")
    parts.append(f"({', '.join(params)})")
    
    # Add return type with proper spacing
    if method.return_type != "()":
        parts.append("->")
        parts.append(method.return_type)
    
    # Format signature
    signature = " ".join(parts)
    # Add return type if it's a Result
    if method.return_type.startswith("Result<"):
        # Extract the success type
        success_type = method.return_type.split("<")[1].split(",")[0].strip().rstrip(">")
        # Replace without duplicating MPIError
        if ", MPIError" in method.return_type:
            signature = signature.replace(f" -> {method.return_type}", f" -> Result<{success_type}, MPIError>")
        else:
            signature = f"{signature.split(' -> ')[0]} -> Result<{success_type}, MPIError>"
    return signature.strip()

def generate_markdown(doc_crate: DocCrate) -> str:
    """
    Generates a Markdown documentation string for the given DocCrate.
    """
    lines = []

    # Crate Title
    lines.append(f"# Crate: **{doc_crate.name}**")
    lines.append("")
    lines.append(doc_crate.description)
    lines.append("")

    # A helper function to recursively document a module
    def document_module(module: DocModule, depth: int = 2):
        """
        Recursively build Markdown for the given module
        """
        # Heading for the module
        heading = "#" * depth
        lines.append(f"{heading} Module: **{module.name}**")
        lines.append("")
        if module.description:
            lines.append(module.description)
            lines.append("")

        # Document module-level attributes
        if module.attributes:
            lines.append("**Module Attributes:**")
            for attr in module.attributes:
                lines.append(f"- `{attr.name}`({attr.args})")
            lines.append("")

        # Document structs
        if module.structs:
            struct_heading = "#" * (depth + 1)
            lines.append(f"{struct_heading} Structs")
            for s in module.structs:
                lines.append(f"**{s.name}** (visibility: `{s.visibility}`)")
                lines.append("")
                lines.append(s.description)
                lines.append("")
                # Attributes
                if s.attributes:
                    lines.append("**Attributes:**")
                    for attr in s.attributes:
                        lines.append(f"- `{attr.name}`({attr.args})")
                    lines.append("")
                # Fields
                lines.append("**Fields:**")
                for field in s.fields:
                    lines.append(f"- **{field.name}**: *{field.type}*  \n\n"
                                 f"  Visibility: `{field.visibility}`  \n\n"
                                 f"  {field.description}")
                lines.append("")
                # Methods
                if s.methods:
                    lines.append("**Methods:**")
                    for m in s.methods:
                        lines.append(f"- **{m.name}**{_doc_method_signature(m)}")
                        lines.append("")
                        lines.append(f"  Visibility: `{m.visibility}`")
                        lines.append("")
                        if m.description:
                            lines.append(f"  {m.description}")
                            lines.append("")
                        if m.examples:
                            lines.append("  **Examples:**")
                            for ex in m.examples:
                                lines.append(f"```rust\n{ex}\n```")
                        lines.append("")
                lines.append("---")

        # Document enums
        if module.enums:
            enum_heading = "#" * (depth + 1)
            lines.append(f"{enum_heading} Enums")
            for e in module.enums:
                lines.append(f"**{e.name}** (visibility: `{e.visibility}`)")
                lines.append("")
                lines.append(e.description)
                lines.append("")
                if e.attributes:
                    lines.append("**Attributes:**")
                    for attr in e.attributes:
                        lines.append(f"- `{attr.name}`({attr.args})")
                    lines.append("")
                # Variants
                lines.append("**Variants:**")
                for v in e.variants:
                    lines.append(f"- **{v.name}**  \n\n"
                                 f"  {v.description}")
                    if v.fields:
                        lines.append(f"  Fields:")
                        for f_ in v.fields:
                            lines.append(f"    - **{f_.name}**: *{f_.type}*  \n\n"
                                         f"      Visibility: `{f_.visibility}`  \n\n"
                                         f"      {f_.description}")
                lines.append("")
                # Enum methods
                if e.methods:
                    lines.append("**Methods:**")
                    for m in e.methods:
                        lines.append(f"- **{m.name}**{_doc_method_signature(m)}")
                        lines.append("")
                        lines.append(f"  Visibility: `{m.visibility}`")
                        lines.append("")
                        if m.description:
                            lines.append(f"  {m.description}")
                            lines.append("")
                        if m.examples:
                            lines.append("  **Examples:**")
                            for ex in m.examples:
                                lines.append(f"```rust\n{ex}\n```")
                        lines.append("")
                lines.append("---")

        # Document free functions or associated free methods in this module
        if module.methods:
            method_heading = "#" * (depth + 1)
            lines.append(f"{method_heading} Free Functions")
            for m in module.methods:
                lines.append(f"- **{m.name}**{_doc_method_signature(m)}")
                lines.append("")
                lines.append(f"  Visibility: `{m.visibility}`")
                lines.append("")
                if m.description:
                    lines.append(f"  {m.description}")
                    lines.append("")
                if m.examples:
                    lines.append("  **Examples:**")
                    for ex in m.examples:
                        lines.append(f"```rust\n{ex}\n```")
                lines.append("")
            lines.append("---")

        # Recursively document submodules
        if module.submodules:
            for sub in module.submodules:
                document_module(sub, depth=depth+1)

    def _doc_method_signature(method: DocMethod) -> str:
        """
        Build a short signature snippet for a method.
        E.g. (param1: Type, param2: Type) -> ReturnType
        """
        params_str = ", ".join(f"{p.name}: {p.type}" for p in method.parameters)
        signature = f"({params_str})"
        if method.return_type:
            signature += f" -> {method.return_type}"
        return signature

    # Document each top-level module in this crate
    for module in doc_crate.modules:
        document_module(module)

    # Join all lines with a newline and return
    return "\n".join(lines)


if __name__ == '__main__':
    main()
