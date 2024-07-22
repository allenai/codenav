import argparse
import datetime
import glob
import importlib
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from queue import Empty
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Type, Union, Mapping

from codenav.agents.gpt4.agent import OpenAICodeNavAgent
from codenav.constants import DEFAULT_RETRIEVAL_PER_QUERY
from codenav.environments.code_env import PythonCodeEnv
from codenav.environments.done_env import DoneEnv
from codenav.environments.retrieval_env import RetrievalEnv
from codenav.interaction.episode import Episode
from codenav.retrieval.elasticsearch.elasticsearch_retriever import EsCodeRetriever
from codenav.utils.logging_utils import StringAsFileArtifact, WandbClient, WandbServer
from codenav.utils.string_utils import str2bool

if sys.platform == "darwin":
    mp = mp.get_context("spawn")  # type: ignore
else:
    mp = mp.get_context("forkserver")  # type: ignore


class Task(ABC):
    @property
    def task_name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def inputs(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def task_template(self) -> str:
        pass

    @property
    def task_description(self) -> str:
        return self.task_template.format(**self.inputs)

    @abstractmethod
    def log_io(self, logger, episode: Optional[Episode] = None):
        raise NotImplementedError

    def log(self, logger, episode: Optional[Episode] = None):
        self.log_io(logger, episode)
        if episode is not None:
            logger.log(
                {
                    f"{self.task_name}/interactions": logger.Table(
                        dataframe=episode.tabulate_interactions()
                    )
                }
            )
            logger.log(
                {
                    f"{self.task_name}/execution_trace": logger.Table(
                        dataframe=episode.tabulate_exec_trace()
                    )
                }
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_names",
        nargs="+",
        type=str,
        help="List of task names to evaluate on.",
    )
    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument("--index_name", type=str, required=True)
    parser.add_argument("--host", type=str, default="http://localhost:9200/")
    parser.add_argument("--reindex", type=str2bool)
    parser.add_argument(
        "--retrievals_per_keyword", type=int, default=DEFAULT_RETRIEVAL_PER_QUERY
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--nprocesses", type=int, default=10)
    return parser.parse_args()


def build_codenav_episode(
    query: str,
    retrievals_per_keyword: int,
    code_dir: str,
    index_name: str,
    host: str,
    working_dir: str,
) -> Episode:
    sys_paths = []

    # Here we check if adding code_dir to the system path would cause namespace conflicts with any existing importable
    # modules. If it would, we assume it has already been installed into the environment and so we don't add it to the
    # PYTHONPATH of the PythonCodeEnv.
    conflicts = []
    if code_dir not in sys.path:
        for p in glob.glob(os.path.join(code_dir, "*")):
            if p.endswith(".py") or (
                os.path.isdir(p) and os.path.exists(os.path.join(p, "__init__.py"))
            ):
                try:
                    importlib.import_module(
                        os.path.basename(p) if os.path.isdir(p) else p[:-3]
                    )
                    conflicts.append(p)
                    break  # Can comment this out if we want a full list of conflicts
                except ImportError:
                    pass

        if len(conflicts) != 0:
            warnings.warn(
                f"Adding {code_dir} to the system path would result in namespace conflicts as the paths {conflicts} are"
                f" already importable. We will assume this means that the codebase has been already been installed"
                f" (e.g. with pip) into the environment and so we will not explicitly add it to the PYTHONPATH"
                " of the PythonCodeEnv."
            )
        else:
            sys_paths.append(code_dir)

    return Episode(
        agent=OpenAICodeNavAgent(retrievals_per_keyword=retrievals_per_keyword),
        action_type_to_env={
            "code": PythonCodeEnv(
                code_dir=code_dir, sys_paths=sys_paths, working_dir=working_dir
            ),
            "search": RetrievalEnv(
                code_retriever=EsCodeRetriever(index_name=index_name, host=host),
                expansions_per_query=retrievals_per_keyword,
                prototypes_per_query=10,
                max_per_query=20,
                summarize_code=True,
            ),
            "done": DoneEnv(),
        },
        user_query_str=query,
    )


def task_runner(
    worker_index: int,
    wandb_client: WandbClient,
    task_info_queue: mp.Queue,
    task_str_to_task_class: Dict[str, Type[Task]],
    code_dir: str,
    index_name: str,
    host: str,
    full_exp_name: str,
    working_dir: str,
    print_steps: bool = True,
) -> bool:
    os.makedirs(working_dir, exist_ok=True)

    try:
        wandb_client.init()
        while True:
            try:
                task_info = task_info_queue.get(timeout=1)
                print(
                    f"[WORKER {worker_index}] Beginning task: {task_info['task_name']}"
                )

                task_class = task_str_to_task_class[task_info["task_name"]]
                if isinstance(task_class, Task):
                    task = task_class
                else:
                    task = task_str_to_task_class[task_info["task_name"]]()

                episode = build_codenav_episode(
                    query=task.task_description,
                    retrievals_per_keyword=task_info["retrievals_per_keyword"],
                    index_name=index_name,
                    host=host,
                    code_dir=code_dir,
                    working_dir=working_dir,
                )

                try:
                    episode.step_until_max_steps_or_success(
                        max_steps=task_info["max_steps"], verbose=print_steps
                    )
                except:
                    print(
                        f"[WORKER {worker_index}] Exception encountered during episode."
                    )
                    print(traceback.format_exc())
                    return False

                print(
                    f"[WORKER {worker_index}] Completed task: {task_info['task_name']}, logging..."
                )

                task.log(logger=wandb_client, episode=episode)

                notebook_name = f"{task_info['task_name']}__{full_exp_name}"
                artifact_info = dict(
                    name=notebook_name,
                    type="notebook",
                    description="Notebook for simple_hf evaluation.",
                )

                wandb_client.log_string_as_file_artifact(
                    StringAsFileArtifact(
                        string=episode.to_notebook(cur_dir=working_dir),
                        file_name=f"{notebook_name}.ipynb",
                        artifact_info=artifact_info,
                    )
                )
            except Empty:
                print(f"[WORKER {worker_index}] No more tasks to run, quitting...")
                break
    finally:
        wandb_client.close()

    return True


def eval_manager(
    exp_prefix: str,
    task_str_to_task_class: Mapping[str, Union[Task, Type[Task]]],
    code_dir: str,
    working_dir: str,
    args: Any,
    project: str = "codenav",
    wandb_online: bool = True,
    wandb_dir: Optional[str] = None,
    **task_runner_kwargs,
):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_exp_name = exp_prefix + "__" + args.exp_name + "__" + date_str

    td = TemporaryDirectory()
    with td as tmp_wandb_dir:
        if wandb_dir is None:
            wandb_dir = tmp_wandb_dir

        wandb_server = WandbServer(
            queue=mp.Queue(),
            project=project,
            entity="prior-ai2",
            name=full_exp_name,
            dir=wandb_dir,
            mode="online" if wandb_online else "offline",
        )

        task_info_queue: "mp.Queue[Dict[str, Union[str, int]]]" = mp.Queue()
        for task_name in args.task_names:
            task_info_queue.put(
                {
                    "task_name": task_name,
                    "retrievals_per_keyword": args.retrievals_per_keyword,
                    "max_steps": args.max_steps,
                }
            )

        nprocesses = min(args.nprocesses, len(args.task_names))

        start = time.time()
        processes = []
        for worker_ind in range(nprocesses):
            processes.append(
                mp.Process(
                    target=task_runner,
                    kwargs=dict(
                        worker_index=worker_ind,
                        wandb_client=wandb_server.create_client(),
                        task_info_queue=task_info_queue,
                        task_str_to_task_class=task_str_to_task_class,
                        code_dir=code_dir,
                        index_name=args.index_name,
                        host=args.host,
                        working_dir=working_dir,
                        full_exp_name=full_exp_name,
                        **task_runner_kwargs,
                    ),
                )
            )
            processes[-1].start()

        while wandb_server.any_open_clients():
            try:
                wandb_server.log(timeout=5, verbose=True)
            except Empty:
                pass

        for p in processes:
            p.join(timeout=1)

        print(f"Total time taken for eval: {time.time() - start}")

        wandb_server.finish()
