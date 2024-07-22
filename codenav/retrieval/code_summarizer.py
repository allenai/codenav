import ast
import re

from codenav.constants import DEFAULT_OPENAI_MODEL, OPENAI_CLIENT
from codenav.utils.llm_utils import create_openai_message, query_gpt
from codenav.utils.string_utils import get_tag_content_from_text

SUMMARIZATION_PROMPT = """\
You are an expert python programmer. Given a user provided code snippet that represents a function or a class definition, you will document that code. Your documentation should be concise as possible while still being sufficient to fully understand how the code behaves. Never mention that the code is being documented by an AI model or that it is in any particular style.

If you are given a class definition which, for example, defines functions func1, func2, ..., funcN, you response should be formatted as:
```  
<ClassName>
Doc string of the class, should include Attributes (if any).
</ClassName>

<func1> 
Google-style doc string of func1, should include Args, Returns, Yields, Raises, and Examples if applicable.
</func1>

(Doc strings for func2, ..., funcN, each in their own <func?></func?> block)
```

If you are given a function called func, then format similarly as:
```
<func>
Doc string of func, should include Args, Returns, Yields, Raises, and Examples if applicable.
</func>
```

Do not include starting or ending ``` in your response. Ensure you document the __init__ function if defined. Your output should start with <ClassName> including the < and > characters.
"""


class DocstringTransformer(ast.NodeTransformer):
    def __init__(self, docstring_map):
        self.docstring_map = docstring_map

    def indent_docstring(self, docstring: str, offset: int = 0):
        indentation = " " * offset
        if "\n" in docstring:
            lines = docstring.split("\n") + [""]
            return "\n" + "\n".join([f"{indentation}{line}" for line in lines])

    def visit_ClassDef(self, node):
        # Update class docstring
        if node.name in self.docstring_map:
            docstring = self.indent_docstring(
                self.docstring_map[node.name], offset=node.col_offset + 4
            )
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Str, ast.Constant))
            ):
                # Replace the existing docstring
                node.body[0] = ast.Expr(value=ast.Str(s=docstring))
            else:
                # Insert new docstring if none exists
                node.body.insert(0, ast.Expr(value=ast.Str(s=docstring)))
            # node.body.insert(0, ast.Expr(value=ast.Str(s=docstring)))
        self.generic_visit(node)  # Process methods
        return node

    def visit_FunctionDef(self, node):
        # Update method docstring and replace body
        if node.name in self.docstring_map:
            docstring = self.indent_docstring(
                self.docstring_map[node.name], offset=node.col_offset + 4
            )
            node.body = [
                ast.Expr(value=ast.Constant(value=docstring)),
                ast.parse("...").body,
            ]
        else:
            node.body = [ast.parse("...").body]
        return node


def rewrite_docstring(code, docstring_map):
    tree = ast.parse(code)
    transformer = DocstringTransformer(docstring_map)
    transformed_tree = transformer.visit(tree)
    return ast.unparse(transformed_tree)


class CodeSummarizer:
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        max_tokens: int = 50000,
    ):
        self.model = model
        self.max_tokens = max_tokens

    def generate_tagged_summary(self, code: str) -> str:
        # Use OpenAI to summarize the code
        messages = [
            create_openai_message(text=SUMMARIZATION_PROMPT, role="system"),
            create_openai_message(text=code, role="user"),
        ]
        return query_gpt(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            client=OPENAI_CLIENT,
        )

    def summarize(self, code: str) -> str:
        tagged_summary = self.generate_tagged_summary(code)
        tags = [t.strip() for t in re.findall(r"<([^/]*?)>", tagged_summary)]
        docstring_map = dict()
        for tag in tags:
            docstring_map[tag] = get_tag_content_from_text(tagged_summary, tag=tag)

        return rewrite_docstring(code, docstring_map)


if __name__ == "__main__":
    import inspect
    import sys
    import importlib

    func_or_class = sys.argv[1]

    m = importlib.import_module(".".join(func_or_class.split(".")[:-1]))
    to_doc = inspect.getsource(getattr(m, func_or_class.split(".")[-1]))

    summarizer = CodeSummarizer()
    print(summarizer.summarize(to_doc))
