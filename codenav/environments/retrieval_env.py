import os
import traceback
from typing import Dict, List, Sequence, Tuple, Union

import codenav.interaction.messages as msg
from codenav.environments.abstractions import CodeNavEnv
from codenav.interaction.messages import CodeNavAction
from codenav.retrieval.code_blocks import CodeBlockType
from codenav.retrieval.elasticsearch.create_index import EsDocument, es_doc_to_hash
from codenav.retrieval.elasticsearch.elasticsearch_retriever import (
    EsCodeRetriever,
    parallel_add_summary_to_es_docs,
)


def reorder_es_docs(es_docs: List[EsDocument]) -> List[EsDocument]:
    scores = []
    for es_doc in es_docs:
        t = es_doc["type"]
        proto = es_doc.get("prototype") or ""

        if t in [
            CodeBlockType.FUNCTION.name,
            CodeBlockType.CLASS.name,
            CodeBlockType.DOCUMENTATION.name,
        ]:
            score = 0.0
        elif t in [CodeBlockType.IMPORT.name, CodeBlockType.ASSIGNMENT.name]:
            score = 1.0
        else:
            score = 2.0

        if (
            os.path.basename(es_doc["file_path"]).lower().startswith("test_")
            or os.path.basename(es_doc["file_path"]).lower().endswith("_test.py")
            or (proto is not None and "test_" in proto.lower() or "Test" in proto)
        ):
            score += 2.0

        scores.append(score)

    return [
        es_doc for _, es_doc in sorted(list(zip(scores, es_docs)), key=lambda x: x[0])
    ]


class RetrievalEnv(CodeNavEnv):
    def __init__(
        self,
        code_retriever: EsCodeRetriever,
        expansions_per_query: int,
        prototypes_per_query: int,
        max_per_query: int = 100,
        summarize_code: bool = True,
        overwrite_existing_summary: bool = False,
    ):
        self.code_retriever = code_retriever
        self.expansions_per_query = expansions_per_query
        self.prototypes_per_query = prototypes_per_query
        self.max_per_query = max_per_query
        self.summarize_code = summarize_code
        self.overwrite_existing_summary = overwrite_existing_summary

        self.retrieved_es_docs: Dict[str, EsDocument] = {}

    def reset(self):
        self.retrieved_es_docs = dict()

    def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
        return True, ""

    def _get_retrieval_result(self, query: str):
        query = query.strip()

        try:
            es_docs = self.code_retriever.search(
                query=query, default_n=self.max_per_query
            )
        except:
            error_msg = traceback.format_exc()
            print(error_msg)
            if "Failed to parse query" in error_msg:
                error_msg = (
                    f"Failed to parse search query: {query}\n"
                    f"Please check the syntax and try again (be careful to escape any reserved characters)."
                )

            return msg.RetrievalResult(
                query=query,
                es_docs=[],
                failure_reason=error_msg,
            )

        filtered_es_docs = [
            es_doc
            for es_doc in es_docs
            if es_doc_to_hash(es_doc) not in self.retrieved_es_docs
        ]

        failure_reason = None
        if len(filtered_es_docs) == 0 and len(es_docs) > 0:
            failure_reason = (
                f"All code blocks matching the query have already been returned."
            )

        filtered_es_docs = reorder_es_docs(filtered_es_docs)

        # todo: add summaries to filtered_es_docs if jit_summarize is True
        if self.summarize_code:
            parallel_add_summary_to_es_docs(
                es_docs=filtered_es_docs[: self.expansions_per_query],
                es=self.code_retriever.es,
                index_name=self.code_retriever.index_name,
                code_summarizer=self.code_retriever.code_summarizer,
                overwrite_existing=self.overwrite_existing_summary,
            )

        for es_doc in filtered_es_docs[: self.expansions_per_query]:
            self.retrieved_es_docs[es_doc_to_hash(es_doc)] = es_doc

        return msg.RetrievalResult(
            query=query,
            es_docs=filtered_es_docs,
            failure_reason=failure_reason,
            max_expanded=self.expansions_per_query,
            max_prototype=self.prototypes_per_query,
        )

    def step(
        self, action: Union[CodeNavAction, str, Sequence[str]]
    ) -> msg.MultiRetrievalResult:
        if isinstance(action, str):
            queries = [q.strip() for q in action.split("\n") if q.strip() != ""]
        elif isinstance(action, CodeNavAction):
            if action.content is None:
                queries = []
            else:
                queries = [
                    q.strip() for q in action.content.split("\n") if q.strip() != ""
                ]
        else:
            queries = [q.strip() for q in action if q.strip() != ""]

        retrieval_results: List[msg.RetrievalResult] = [
            self._get_retrieval_result(query=query) for query in queries
        ]

        return msg.MultiRetrievalResult(retrieval_results=retrieval_results)
