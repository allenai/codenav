import ast
import os
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set


class CodeBlockType(Enum):
    ASSIGNMENT = 1
    FUNCTION = 2
    CLASS = 3
    FILE = 4
    IMPORT = 5
    CONDITIONAL = 6
    DOCUMENTATION = 7
    OTHER = 8


AST_TYPE_TO_CODE_BLOCK_TYPE = {
    ast.Module: CodeBlockType.FILE,
    ast.FunctionDef: CodeBlockType.FUNCTION,
    ast.ClassDef: CodeBlockType.CLASS,
    ast.Assign: CodeBlockType.ASSIGNMENT,
    ast.Import: CodeBlockType.IMPORT,
    ast.ImportFrom: CodeBlockType.IMPORT,
    ast.If: CodeBlockType.CONDITIONAL,
    ast.For: CodeBlockType.CONDITIONAL,
}


CODE_BLOCK_TEMPLATE = """file_path={file_path},
lines=[{start_lineno}, {end_lineno}],
type={type},
content={{
{code}
}}"""


DEFAULT_CODEBLOCK_TYPES = {
    CodeBlockType.FUNCTION,
    CodeBlockType.CLASS,
    CodeBlockType.ASSIGNMENT,
    CodeBlockType.IMPORT,
    CodeBlockType.CONDITIONAL,
    CodeBlockType.DOCUMENTATION,
}


class FilePath:
    def __init__(self, path: str, base_dir: str = "/"):
        self.path = os.path.abspath(path)
        self.base_dir = base_dir

    @property
    def file_name(self) -> str:
        """root/base_dir/path/to/file.py -> file.py"""
        return os.path.basename(self.path)

    @property
    def ext(self) -> str:
        """root/base_dir/path/to/file.py -> .py"""
        return os.path.splitext(self.path)[1]

    @property
    def rel_path(self) -> str:
        """root/base_dir/path/to/file.py -> path/to/file.py"""
        return os.path.relpath(self.path, self.base_dir)

    @property
    def import_path(self) -> str:
        """root/base_dir/path/to/file.py -> base_dir/path/to/file.py"""
        return os.path.join(os.path.basename(self.base_dir), self.rel_path)

    @property
    def abs_path(self) -> str:
        """root/base_dir/path/to/file.py -> /root/base_dir/path/to/file.py"""
        return self.path

    def __repr__(self) -> str:
        """FilePath(base_dir=root/base_dir, rel_path=path/to/file.py)"""
        return f"FilePath(base_dir={self.base_dir}, rel_path={self.rel_path})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FilePath):
            return NotImplemented
        return self.path == other.path and self.base_dir == other.base_dir

    def __hash__(self):
        return hash(str(self))


def get_file_list(dir_path: str) -> list[FilePath]:
    file_list = []
    for root, dirs, files in os.walk(dir_path, followlinks=True):
        for file in files:
            file_list.append(FilePath(path=os.path.join(root, file), base_dir=dir_path))
    return file_list


def filter_by_extension(
    file_list: list[FilePath], valid_extensions: Iterable[str] = (".py",)
) -> list[FilePath]:
    return [file_path for file_path in file_list if file_path.ext in valid_extensions]


class CodeBlockASTNode:
    def __init__(
        self,
        ast_node: ast.AST,
        parent: Optional["CodeBlockASTNode"] = None,
        tree: Optional["CodeBlockAST"] = None,
    ):
        self.ast_node = ast_node
        self.parent = parent
        self.tree = tree
        self.code_summary: Optional[str] = None

    @staticmethod
    def format_code_block(
        file_path: str,
        start_lineno: Optional[int],
        end_lineno: Optional[int],
        type: str,
        code: str,
    ) -> str:
        return CODE_BLOCK_TEMPLATE.format(
            file_path=file_path,
            start_lineno=start_lineno,
            end_lineno=end_lineno,
            type=type,
            code=code,
        )

    def __repr__(self) -> str:
        if self.tree is None or self.tree.file_path is None:
            file_path = "None"
        else:
            file_path = self.tree.file_path.rel_path

        return self.format_code_block(
            file_path=file_path,
            start_lineno=self.ast_node.lineno - 1,
            end_lineno=self.ast_node.end_lineno,
            type=self.block_type.name,
            code=self.code if self.code_summary is None else self.code_summary,
        )

    @property
    def block_type(self) -> CodeBlockType:
        return AST_TYPE_TO_CODE_BLOCK_TYPE.get(type(self.ast_node), CodeBlockType.OTHER)

    @property
    def code(self):
        return "\n".join(
            self.tree.code_lines[self.ast_node.lineno - 1 : self.ast_node.end_lineno]
        )

    def children(
        self, included_types: Optional[Set[CodeBlockType]] = None
    ) -> List["CodeBlockASTNode"]:
        if self.tree is None:
            raise RuntimeError(
                "Cannot call children on a node that is not part of a tree."
            )

        return [
            self.tree.ast_node_to_node[child]
            for child in ast.iter_child_nodes(self.ast_node)
            if included_types is None
            or self.tree.ast_node_to_node[child].block_type in included_types
        ]


class CodeBlockAST:
    def __init__(self, code: str, file_path: Optional[FilePath]):
        self.code = code
        self.code_lines = code.splitlines()
        self.file_path = file_path
        self.ast_root = ast.parse(self.code)
        self.ast_node_to_node: Dict[ast.AST, "CodeBlockASTNode"] = {}
        self._build_tree(
            CodeBlockASTNode(
                ast_node=self.ast_root,
                parent=None,
                tree=self,
            )
        )

    @property
    def root(self) -> CodeBlockASTNode:
        return self.ast_node_to_node[self.ast_root]

    @staticmethod
    def from_file_path(file_path: FilePath) -> "CodeBlockAST":
        with open(file_path.path, "r") as f:
            code = f.read()

        return CodeBlockAST(
            code=code,
            file_path=file_path,
        )

    def _build_tree(self, node: CodeBlockASTNode):
        self.ast_node_to_node[node.ast_node] = node
        for child in ast.iter_child_nodes(node.ast_node):
            child_node = CodeBlockASTNode(ast_node=child, parent=node, tree=self)
            self._build_tree(child_node)

    def __repr__(self) -> str:
        return ast.dump(self.ast_root)
