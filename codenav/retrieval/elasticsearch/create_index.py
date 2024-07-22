from typing import Optional, TypedDict, Union, Literal

import tiktoken
from elasticsearch import Elasticsearch

from codenav.utils.hashing_utils import md5_hash_str


class EsDocument(TypedDict):
    text: str
    file_path: str
    lines: dict
    type: str  # Will be string corresponding to one of the CodeBlockType types.
    prototype: str
    text_vector: Optional[list[float]]
    text_summary: Optional[str]


def es_doc_to_string(
    doc: EsDocument,
    prototype: bool = False,
    use_summary: Union[bool, Literal["ifshorter", "always", "never"]] = "ifshorter",
) -> str:
    if prototype:
        return doc["prototype"] + " # " + doc["file_path"]

    code: Optional[str] = doc["text"]
    summary = doc.get("text_summary", code)

    if isinstance(use_summary, str):
        if use_summary == "ifshorter":
            encode_count = lambda m: len(
                tiktoken.get_encoding("cl100k_base").encode(m, disallowed_special=[])
            )

            use_summary = encode_count(code) >= encode_count(summary)
        elif use_summary == "always":
            use_summary = True
        elif use_summary == "never":
            use_summary = False
        else:
            raise ValueError("Invalid value for use_summary")

    doc_str = [
        f"file_path={doc['file_path']}",
        f"lines=[{doc['lines']['gte']}, {doc['lines']['lt']}]",
        f"type={doc['type']}",
        f"content={{\n{summary if use_summary else code}\n}}",
    ]

    return "\n".join(doc_str)


def es_doc_to_hash(doc: EsDocument) -> str:
    return md5_hash_str(es_doc_to_string(doc, prototype=False, use_summary=False))


class EsHit(TypedDict):
    _index: str
    _type: str
    _id: str
    _score: float
    _source: EsDocument


CODENAV_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "tokenizer": {
                "code_ngram_3_4_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 4,
                    "token_chars": ["letter", "digit", "punctuation", "symbol"],
                }
            },
            "analyzer": {
                "code_ngram_3_4_analyzer": {
                    "type": "custom",
                    "tokenizer": "code_ngram_3_4_tokenizer",
                    "filter": ["lowercase"],  # You can include other filters as needed
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "file_path": {"type": "keyword"},
            "type": {"type": "keyword"},
            "lines": {"type": "integer_range"},  # Using integer_range for line numbers
            "text": {
                "type": "text",
                "analyzer": "code_ngram_3_4_analyzer",
            },
            "prototype": {
                "type": "text",
                "analyzer": "code_ngram_3_4_analyzer",
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 1536,  # Adjust the dimension according to your vector
                "index": "true",
                "similarity": "cosine",
            },
            "text_summary": {
                "type": "text",
                "analyzer": "code_ngram_3_4_analyzer",
            },
        }
    },
}


def create_empty_index(
    es: Elasticsearch,
    index_name: str,
    embedding_dim: int,
):
    assert embedding_dim == 1536, "Only 1536-dimensional vectors are supported"
    # Create the index
    es.indices.create(
        index=index_name, body=CODENAV_INDEX_SETTINGS, request_timeout=120
    )


def create_file_path_hash_index(es: Elasticsearch, index_name: str):
    if es.indices.exists(index=index_name):
        return

    # Define the index settings and mappings
    index_body = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "md5hash": {"type": "keyword"},
                "code_uuid": {"type": "keyword"},
                "file_path": {"type": "keyword"},
                "doc_count": {"type": "integer"},
            }
        },
    }

    # Create the index
    es.indices.create(index=index_name, body=index_body)


if __name__ == "__main__":
    # Connect to the local Elasticsearch instance
    es = Elasticsearch(hosts="http://localhost:9200/")

    create_empty_index(es=es, index_name="code_repository", embedding_dim=1536)

    # Example document
    doc = {
        "text": "def example_function():\n    pass",
        "file_path": "/path/to/file.py",
        "lines": {"gte": 5, "lte": 30},
        "type": "function",
        "text_vector": [0.1] * 768,  # Example vector, replace with actual data
    }

    # Index the document
    es.index(index="code_repository", document=doc)

    query = {"query": {"match": {"type": "function"}}}

    res = es.search(index="code_repository", body=query)
    print(res["hits"]["hits"])
