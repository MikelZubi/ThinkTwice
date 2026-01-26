import inspect
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
import re

def class_to_markdown(cls: Type[BaseModel], output_file: Optional[str] = None) -> str:
    """
    Convert a Pydantic class's documentation to Markdown format.
    
    Args:
        cls: The Pydantic class to document
        output_file: Optional file path to save the markdown. If None, returns the string
    
    Returns:
        str: The markdown documentation
    """
    if not issubclass(cls, BaseModel):
        raise ValueError("Class must be a Pydantic BaseModel")
    
    # Get the class name and prepare markdown content
    markdown_content = [
        f"## {cls.__name__}",
        "### Fields",
        ""
    ]
    
    # Get all fields from the model
    for field_name, field in cls.model_fields.items():
        # Skip private fields
        if field_name.startswith('_') or field_name == 'template_type':
            continue
        
        # Get field type annotation
        field_type = str(field.annotation).replace('typing.', '')
            
        # Get field description from Field parameters
        description = field.description or "No description available."
        
        # Get default value if any
        default_value = field.default
        default_str = ""
        if default_value is not None:
            if isinstance(default_value, str):
                default_str = f' (default: "{default_value}")'
            else:
                default_str = f' (default: {default_value})'
        
        # Add field documentation
        markdown_content.extend([
            f"#### `{field_name}`",
            "",
            f"**Type:** `{field_type}`{default_str}".replace("BETTER_Granular_Class_simplified.",""),
            "",
            description,
            ""
        ])
        
        # If the field has a literal constraint, document the allowed values
        '''
        if hasattr(field.annotation, '__args__'):
            args = field.annotation.__args__
            if all(isinstance(arg, str) for arg in args if arg is not type(None)):
                literal_values = [arg for arg in args if arg is not type(None)]
                if literal_values:
                    markdown_content.extend([
                        "**Allowed values:**",
                        "",
                        "- " + "\n- ".join([f'"{value}"' for value in literal_values]),
                        ""
                    ])
        '''
    
    # Join all lines with newlines
    final_markdown = "\n".join(markdown_content)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(final_markdown)
            
    return final_markdown

# Example usage
if __name__ == "__main__":
    # Import your classes here
    import sys
    sys.path.append("class_data")
    #from BETTER_Granular_Class_simplified import span, event, Protestplate, Corruplate, Terrorplate, Epidemiplate, Disasterplate, Displacementplate
    from BETTER_Granular_Class_string import Protestplate, Corruplate, Terrorplate, Epidemiplate, Disasterplate, Displacementplate
    import MUC_Class
    import MUC_Class_simplified
    
    # Generate documentation for each class
    for cls in [Protestplate, Corruplate, Terrorplate, Epidemiplate, Disasterplate, Displacementplate]:
        markdown = class_to_markdown(cls, f"Docs/BETTER/{cls.__name__}.md")
    #_ = class_to_markdown(MUC_Class.Template, f"Docs/MUC.md")
    #_ = class_to_markdown(MUC_Class_simplified.Template, f"Docs/MUC_simplified.md")