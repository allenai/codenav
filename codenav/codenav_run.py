import importlib
import os
import re
import subprocess
import time
from argparse import ArgumentParser
from typing import Optional, Dict, Any, Sequence

import attrs
from elasticsearch import Elasticsearch

from codenav.interaction.episode import Episode
from codenav.retrieval.elasticsearch.index_codebase import (
    DEFAULT_ES_HOST,
    DEFAULT_ES_PORT,
    DEFAULT_KIBANA_PORT,
    build_index,
)
from codenav.retrieval.elasticsearch.install_elasticsearch import (
    is_es_installed,
    install_elasticsearch,
    ES_PATH,
    KIBANA_PATH,
)
from codenav.utils.eval_utils import eval_manager, Task


def is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def is_es_running():
    es = Elasticsearch(DEFAULT_ES_HOST)
    return es.ping()


def run_init():
    es = Elasticsearch(DEFAULT_ES_HOST)
    if es.ping():
        print(
            "Initialization complete, Elasticsearch is already running at http://localhost:9200."
        )
        return

    if not is_es_installed():
        print("Elasticsearch installation not found, downloading...")
        install_elasticsearch()

    if not is_es_installed():
        raise ValueError("Elasticsearch installation failed")

    if is_port_in_use(DEFAULT_ES_PORT) or is_port_in_use(DEFAULT_KIBANA_PORT):
        raise ValueError(
            f"The ports {DEFAULT_ES_PORT} and {DEFAULT_KIBANA_PORT} are already in use,"
            f" to start elasticsearch we require that these ports are free."
        )

    cmd = os.path.join(ES_PATH, "bin", "elasticsearch")
    print(f"Starting Elasticsearch server with command: {cmd}")
    es_process = subprocess.Popen(
        [cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    cmd = os.path.join(KIBANA_PATH, "bin", "kibana")
    print(f"Starting Kibana server with command: {cmd}")
    kibana_process = subprocess.Popen(
        [cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    es_started = False
    kibana_started = False
    try:
        for _ in range(10):
            es_started = es.ping()
            kibana_started = is_port_in_use(DEFAULT_KIBANA_PORT)

            if not es_started:
                print("Elasticsearch server not started yet...")

            if not kibana_started:
                print("Kibana server not started yet...")

            if es_started and kibana_started:
                break

            print("Waiting 10 seconds...")
            time.sleep(10)

        if not (es_started and kibana_started):
            raise RuntimeError("Elasticsearch failed to start")

    finally:
        if not (es_started and kibana_started):
            es_process.kill()
            kibana_process.kill()

    # noinspection PyUnreachableCode
    print(
        f"Initialization complete. "
        f" Elasticsearch server started successfully (PID {es_process.pid}) and can be accessed at {DEFAULT_ES_PORT}."
        f" You can also access the Kibana dashboard (PID {kibana_process.pid}) at {DEFAULT_KIBANA_PORT}."
        f" You will need to manually stop these processes when you are done with them."
    )


class QueryCodeNavTask(Task):
    def __init__(self, query: str):
        self.query = query

    @property
    def inputs(self) -> Dict[str, Any]:
        return {}

    @property
    def task_template(self) -> str:
        return self.query

    def log_io(self, logger, episode: Optional[Episode] = None):
        pass


@attrs.define
class EvalManagerConfig:
    exp_name: str
    task_names: Sequence[str]
    retrievals_per_keyword: int
    max_steps: int
    nprocesses: int
    index_name: str
    host: str


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "command",
        help="command to be executed",
        choices=["init", "stop", "query"],
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=None,
        help="Path to the codebase to use. Only one of `code_dir` or `module` should be provided.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Module to use for the codebase. Only one of `code_dir` or `module` should be provided.",
    )

    parser.add_argument(
        "--playground_dir",
        type=str,
        default=None,
        help="The working directory for the agent.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum number of rounds of interaction.",
    )
    parser.add_argument(
        "-q",
        "--q",
        "--query",
        type=str,
        help="A description of the problem you want the the agent to solve (using `code_dir`).",
    )
    parser.add_argument(
        "-f",
        "--query_file",
        type=str,
        default=None,
        help="A path to a file containing your query (useful for long/detailed queries that are hard to enter on the commandline).",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=os.getcwd(),
        help="Directory into which to save logs. If 'None', logs are written to a temporary"
        " directory that will be deleted after the run.",
    )
    parser.add_argument(
        "--force_reindex",
        store_true=True,
        help="Will delete the existing index (if any) and refresh it.",
    )

    args = parser.parse_args()

    if args.wandb_dir.strip().lower() == "none":
        args.wandb_dir = None

    if args.command == "init":
        run_init()
    elif args.command == "stop":
        # Find all processes that start with ES_PATH and KIBANA_PATH and kill them
        for path in [ES_PATH, KIBANA_PATH]:
            cmd = f"ps aux | grep {path} | grep -v grep | awk '{{print $2}}' | xargs kill "
            subprocess.run(cmd, shell=True)
    elif args.command == "query":
        if not is_es_running():
            raise ValueError(
                "Elasticsearch not running, please run `codenav init` first."
            )

        assert (args.q is None) != (
            args.query_file is None
        ), "Exactly one of `q` or `query_file` should be provided"

        if args.query_file is not None:
            print(args.query_file)
            with open(args.query_file, "r") as f:
                args.q = f.read()

        if args.q is None:
            raise ValueError("No query provided")

        if args.code_dir is None == args.module is None:
            raise ValueError("Exactly one of `code_dir` or `module` should be provided")

        if args.code_dir is None and args.module is None:
            raise ValueError("No code_dir or module provided")

        if args.playground_dir is None:
            raise ValueError("No playground_dir provided")

        if args.code_dir is None:
            path_to_module = os.path.abspath(
                os.path.dirname(importlib.import_module(args.module).__file__)
            )
            args.code_dir = os.path.dirname(path_to_module)

            force_subdir = os.path.basename(path_to_module)
            code_name = force_subdir
        else:
            force_subdir = None
            args.code_dir = os.path.abspath(args.code_dir)
            code_name = os.path.basename(args.code_dir)

        args.playground_dir = os.path.abspath(args.playground_dir)

        if args.force_reindex or not Elasticsearch(DEFAULT_ES_HOST).indices.exists(
            index=code_name
        ):
            print(f"Index {code_name} not found, creating index...")
            build_index(
                code_dir=args.code_dir,
                index_uid=code_name,
                delete_index=args.force_reindex,
                force_subdir=force_subdir,
            )

        conf = EvalManagerConfig(
            exp_name=re.sub("[^A-Za-z0–9 ]", "", args.q).replace(" ", "_")[:30],
            task_names=[code_name],
            retrievals_per_keyword=3,
            max_steps=args.max_steps,
            nprocesses=1,
            index_name=code_name,
            host=DEFAULT_ES_HOST,
        )
        eval_manager(
            exp_prefix=code_name,
            task_str_to_task_class={
                code_name: QueryCodeNavTask(args.q),
            },
            code_dir=args.code_dir,
            working_dir=args.playground_dir,
            args=conf,
            print_steps=True,
            wandb_online=False,
            project="codenav",
            wandb_dir=args.wandb_dir,
        )

    else:
        raise ValueError(f"Unrecognized command: {args.command}")


if __name__ == "__main__":
    main()
