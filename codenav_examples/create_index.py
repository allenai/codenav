import os
import sys

import pandas as pd
from elasticsearch import Elasticsearch

from codenav.environments.retrieval_env import RetrievalEnv
from codenav.retrieval.elasticsearch.elasticsearch_retriever import EsCodeRetriever
from codenav.retrieval.elasticsearch.index_codebase import DEFAULT_ES_HOST, build_index

print(f"Looking for a running Elasticsearch server at {DEFAULT_ES_HOST}...")
es = Elasticsearch(DEFAULT_ES_HOST)
if es.ping():
    print(f"Elasticsearch server is running at {DEFAULT_ES_HOST}")
else:
    print(
        f"Elasticsearch server not found at {DEFAULT_ES_HOST}\n"
        "\tStart the server before running this script\n"
        "\tTo start the Elasticsearch server, run `condenav init` or `python -m condenav.codenav_run init`"
    )
    sys.exit(1)


CODE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SUBDIR = "codenav"
print(f"Building index from subdir='{SUBDIR}' of code_dir='{CODE_DIR}'  ...")
build_index(
    code_dir=CODE_DIR,
    force_subdir=SUBDIR,
    delete_index=True,
    index_uid="codenav",
    host=DEFAULT_ES_HOST,
)

print("Creating EsCodeRetriever which can search the index...")
es_code_retriever = EsCodeRetriever(index_name="codenav", host=DEFAULT_ES_HOST)

print("Searching the ES index using `prototype: CodeEnv`...")
search_query = "prototype: CodeSummaryEnv"
raw_search_res = es_code_retriever.search(search_query)

raw_search_res_df = pd.DataFrame.from_records(raw_search_res)
print(raw_search_res_df)

print("Creating retrieval environment that adds state logic...")
env = RetrievalEnv(
    code_retriever=es_code_retriever,
    expansions_per_query=3,
    prototypes_per_query=5,
    summarize_code=False,
)
response = env.step(search_query)
print(response.format())
