import ast
from _ast import AST
from typing import Union, cast


def get_class_or_function_prototype(
    code: Union[str, ast.ClassDef, ast.FunctionDef],
    include_init: bool = True,
) -> str:
    """
    Summarizes the given Python class or function definition code.

    For classes by, this will occur by keeping the class name and its __init__ method signature,
    replacing the body of __init__ with an ellipsis (...).

    For functions, this will occur by keeping the function name and its signature, replacing the body with an ellipsis (...).

    Args:
    - code: A string containing the Python class definition.

    Returns:
    - A summary string of the form "class ClassName(BaseClass):   def __init__(self, arg1: Type, arg2: Type, ...): ..."
        for a class definition, or "def function_name(arg1: Type, arg2: Type, ...) -> Type: ..." for a function definition.
    """
    if isinstance(code, str):
        tree = ast.parse(code)
        func_or_class = [
            node
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        assert (
            len(func_or_class) == 1
        ), "The given code should contain exactly one class or function definition."
        node = func_or_class[0]
    else:
        node = code

    summary = ""
    if isinstance(node, ast.ClassDef):
        # Format class definition
        base_classes = [ast.unparse(base) for base in node.bases]
        class_header = f"class {node.name}({', '.join(base_classes)}):"
        summary += class_header

        if include_init:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # Format __init__ method signature
                    init_signature = ast.unparse(item.args)
                    summary += f"   def __init__({init_signature}): ..."
                    break
        else:
            summary += " ..."
    elif isinstance(node, ast.FunctionDef):
        # Format function definition
        function_signature = ast.unparse(node.args)

        try:
            return_suff = f" -> {ast.unparse(cast(AST, node.returns))}"
        except:
            return_suff = ""

        summary += f"def {node.name}({function_signature}){return_suff}: ..."

    return summary
