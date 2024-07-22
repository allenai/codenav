import argparse
import os.path
import time
from typing import List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from codenav.retrieval.code_blocks import (
    CodeBlockAST,
    CodeBlockType,
    filter_by_extension,
    get_file_list,
)
from codenav.retrieval.elasticsearch.create_index import (
    EsDocument,
    create_empty_index,
    es_doc_to_hash,
)
from codenav.utils.llm_utils import num_tokens_from_string
from codenav.utils.parsing_utils import get_class_or_function_prototype

DEFAULT_ES_PORT = 9200
DEFAULT_KIBANA_PORT = 5601
DEFAULT_ES_HOST = f"http://localhost:{DEFAULT_ES_PORT}"
DEFAULT_KIBANA_HOST = f"http://localhost:{DEFAULT_KIBANA_PORT}"


def parse_args():
    parser = argparse.ArgumentParser(description="Index codebase")
    parser.add_argument(
        "--code_dir",
        type=str,
        required=True,
        help="Path to the codebase to index",
    )
    parser.add_argument(
        "--index_uid",
        type=str,
        required=True,
        help="Unique identifier for the index",
    )
    parser.add_argument(
        "--delete_index",
        action="store_true",
        help="Delete the index if it already exists",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_ES_HOST,
        help="Elasticsearch host",
    )
    parser.add_argument(
        "--force_subdir",
        type=str,
        default=None,
        help="If provided, only index files in this subdirectory of code_dir. Their path in the index will still be the"
        " path relative to code_dir.",
    )
    return parser.parse_args()


def split_markdown_documentation_into_parts(
    docstring: str, min_tokens: int = 100, split_after_ntokens: int = 1000
) -> Tuple[List[str], List[Tuple[int, int]]]:
    parts: List[str] = []
    line_nums: List[Tuple[int, int]] = []
    current_lines: List[str] = []
    current_line_nums: List[int] = []

    lines = docstring.split("\n")
    for cur_line_idx, line in enumerate(lines):
        line = line.rstrip()

        if line == "" and len(current_lines) == 0:
            # Skip leading empty lines
            continue

        if len(line) == 0:
            current_lines.append("")
            current_line_nums.append(cur_line_idx)
            continue

        if (
            line.lstrip().startswith("#")
            and (len(line) == 1 or line[1] != "#")
            and (num_tokens_from_string("\n".join(current_lines)) > min_tokens)
        ):
            # We're at a top-level header, and we have enough tokens to split
            parts.append("\n".join(current_lines))
            line_nums.append((current_line_nums[0], current_line_nums[-1]))
            current_lines = []
            current_line_nums = []

        current_lines.append(line)
        current_line_nums.append(cur_line_idx)

        # We should split if we're at the end of the document or if we've reached the token limit
        if (
            len(lines) - 1 == cur_line_idx
            or num_tokens_from_string("\n".join(current_lines)) > split_after_ntokens
        ):
            parts.append("\n".join(current_lines))
            line_nums.append((current_line_nums[0], current_line_nums[-1]))
            current_lines = []
            current_line_nums = []

    return parts, line_nums


def _should_skip_python_file(file_path: str):
    with open(file_path, "r") as f:
        first_line = f.readline().strip("\n #").lower()

    if first_line.startswith("index:"):
        return first_line.split(":")[1].strip().lower() == "false"

    return False


def get_es_docs(code_dir: str, force_subdir: Optional[str]):
    all_files = get_file_list(code_dir)
    docs = []

    if force_subdir is not None:
        force_subdir = os.path.abspath(os.path.join(code_dir, force_subdir))
        if force_subdir[-1] != "/":
            force_subdir += "/"

        all_files = [
            file_path
            for file_path in all_files
            if file_path.abs_path.startswith(force_subdir)
        ]

    python_files = filter_by_extension(all_files, valid_extensions=[".py"])
    for file_path in python_files:
        if _should_skip_python_file(file_path.abs_path):
            print(f"Skipping {file_path} as it has `index: false` in the first line.")
            continue

        try:
            code_block_ast = CodeBlockAST.from_file_path(file_path)
        except SyntaxError:
            print(f"Syntax error in {file_path}, skipping...")
            continue

        for block in code_block_ast.root.children():
            # noinspection PyTypeChecker
            docs.append(
                EsDocument(
                    file_path=file_path.rel_path,
                    type=block.block_type.name,
                    lines=dict(
                        gte=block.ast_node.lineno - 1, lt=block.ast_node.end_lineno
                    ),
                    text=block.code,
                    prototype=get_class_or_function_prototype(
                        block.ast_node, include_init=False
                    ),
                )
            )

    doc_files = filter_by_extension(all_files, valid_extensions=[".md"])
    for file_path in doc_files:
        with open(file_path.abs_path, "r") as f:
            for part, line_nums in zip(
                *split_markdown_documentation_into_parts(f.read())
            ):
                docs.append(
                    EsDocument(
                        file_path=file_path.rel_path,
                        type=CodeBlockType.DOCUMENTATION.name,
                        lines=dict(gte=line_nums[0], lt=line_nums[1] + 1),
                        text=part,
                    )
                )

    return docs


def build_index(
    code_dir: str,
    index_uid: str,
    delete_index: bool,
    host: str = DEFAULT_ES_HOST,
    force_subdir: Optional[str] = None,
):
    code_dir = os.path.abspath(code_dir)
    print(f"Indexing codebase at {code_dir} with index_uid {index_uid}")

    assert os.path.exists(code_dir), f"{code_dir} does not exist"

    docs = get_es_docs(code_dir, force_subdir=force_subdir)
    bulk_insert = [
        {
            "_op_type": "index",
            "_index": index_uid,
            "_id": es_doc_to_hash(doc),
            "_source": doc,
        }
        for doc in docs
    ]

    if delete_index:
        es = Elasticsearch(host)
        if es.indices.exists(index=index_uid):
            print("Deleting existing index...")
            es.indices.delete(index=index_uid)

    assert len(bulk_insert) > 0, f"No documents to index in {code_dir}."

    print(f"Indexing {len(bulk_insert)} documents...")
    es = Elasticsearch(host)
    create_empty_index(es=es, index_name=index_uid, embedding_dim=1536)
    bulk(es, bulk_insert)

    time.sleep(2)  # to allow for indexing to finish


if __name__ == "__main__":
    args = parse_args()
    build_index(
        code_dir=args.code_dir,
        index_uid=args.index_uid,
        delete_index=args.delete_index,
        host=args.host,
        force_subdir=args.force_subdir,
    )
