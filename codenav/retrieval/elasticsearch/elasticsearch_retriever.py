from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from elasticsearch import Elasticsearch

from codenav.retrieval.code_blocks import CodeBlockType
from codenav.retrieval.code_summarizer import CodeSummarizer
from codenav.retrieval.elasticsearch.create_index import EsDocument, es_doc_to_hash


class EsCodeRetriever:
    def __init__(self, index_name: str, host: str):
        self.index_name = index_name
        assert (
            index_name is not None and index_name != ""
        ), "Index name cannot be empty."

        self.host = host
        self.es = Elasticsearch(hosts=host)

        if not self.es.ping():
            raise ValueError(
                f"Elasticsearch is not running or could not be reached at {host}."
            )

        self.code_summarizer = CodeSummarizer()

    def search(self, query: str, default_n: int = 10) -> List[EsDocument]:
        body = {"query": {"query_string": {"query": query}}, "size": default_n}
        hits = self.es.search(index=self.index_name, body=body)["hits"]["hits"]
        return [hit["_source"] for hit in hits]


def add_summary_to_es_doc(
    es_doc: EsDocument,
    es: Elasticsearch,
    index_name: str,
    code_summarizer: CodeSummarizer,
    overwrite_existing: bool = False,
):
    if es_doc["type"] == CodeBlockType.DOCUMENTATION.name:
        return

    if "text_summary" in es_doc and es_doc["text_summary"] is not None:
        if not overwrite_existing:
            return

    summary = code_summarizer.summarize(es_doc["text"])
    es_doc["text_summary"] = summary
    es.update(
        index=index_name,
        id=es_doc_to_hash(es_doc),
        body={"doc": {"text_summary": summary}},
    )


def parallel_add_summary_to_es_docs(
    es_docs: List[EsDocument],
    es: Elasticsearch,
    index_name: str,
    code_summarizer: CodeSummarizer,
    overwrite_existing: bool = False,
):
    n = len(es_docs)
    with ThreadPoolExecutor() as executor:
        executor.map(
            add_summary_to_es_doc,
            es_docs,
            [es] * n,
            [index_name] * n,
            [code_summarizer] * n,
            [overwrite_existing] * n,
        )
