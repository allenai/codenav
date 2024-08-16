import os
from typing import Any, List, Optional

from codenav.agents.gpt4.agent import OpenAICodeNavAgent
from codenav.constants import ABS_PATH_OF_CODENAV_DIR, DEFAULT_OPENAI_MODEL
from codenav.environments.code_env import PythonCodeEnv
from codenav.environments.done_env import DoneEnv
from codenav.environments.retrieval_env import EsCodeRetriever, RetrievalEnv
from codenav.interaction.episode import Episode
from codenav.retrieval.elasticsearch.elasticsearch_constants import RESERVED_CHARACTERS
from codenav.retrieval.elasticsearch.index_codebase import DEFAULT_ES_HOST
from codenav.utils.eval_types import EvalInput, EvalSpec, Str2AnyDict
from codenav.utils.evaluator import CodenavEvaluator
from codenav.utils.prompt_utils import PROMPTS_DIR, PromptBuilder


class DefaultEvalSpec(EvalSpec):
    def __init__(
        self,
        episode_kwargs: Str2AnyDict,
        interaction_kwargs: Str2AnyDict,
        logging_kwargs: Str2AnyDict,
    ):
        super().__init__(episode_kwargs, interaction_kwargs, logging_kwargs)

    @staticmethod
    def build_episode(
        eval_input: EvalInput,
        episode_kwargs: Optional[Str2AnyDict] = None,
    ) -> Episode:
        assert episode_kwargs is not None

        prompt_builder = PromptBuilder(
            repo_description=episode_kwargs["repo_description"]
        )
        prompt = prompt_builder.build(
            dict(
                AVAILABLE_ACTIONS=episode_kwargs["allowed_actions"],
                RESERVED_CHARACTERS=RESERVED_CHARACTERS,
                RETRIEVALS_PER_KEYWORD=episode_kwargs["retrievals_per_keyword"],
            )
        )

        return Episode(
            agent=OpenAICodeNavAgent(
                prompt=prompt,
                model=episode_kwargs["llm"],
                allowed_action_types=episode_kwargs["allowed_actions"],
            ),
            action_type_to_env=dict(
                code=PythonCodeEnv(
                    code_dir=episode_kwargs["code_dir"],
                    sys_paths=episode_kwargs["sys_paths"],
                    working_dir=episode_kwargs["working_dir"],
                ),
                search=RetrievalEnv(
                    code_retriever=EsCodeRetriever(
                        index_name=episode_kwargs["index_name"],
                        host=episode_kwargs["host"],
                    ),
                    expansions_per_query=episode_kwargs["retrievals_per_keyword"],
                    prototypes_per_query=episode_kwargs["prototypes_per_keyword"],
                    summarize_code=False,
                ),
                done=DoneEnv(),
            ),
            user_query_str=eval_input.query,
        )

    @staticmethod
    def run_interaction(
        episode: Episode,
        interaction_kwargs: Optional[Str2AnyDict] = None,
    ) -> Str2AnyDict:
        assert interaction_kwargs is not None
        episode.step_until_max_steps_or_success(
            max_steps=interaction_kwargs["max_steps"],
            verbose=interaction_kwargs["verbose"],
        )
        ipynb_str = episode.to_notebook(cur_dir=episode.code_env.working_dir)
        return dict(ipynb_str=ipynb_str)

    @staticmethod
    def log_output(
        interaction_output: Str2AnyDict,
        eval_input: EvalInput,
        logging_kwargs: Optional[Str2AnyDict] = None,
    ) -> Any:
        assert logging_kwargs is not None

        outfile = os.path.join(logging_kwargs["out_dir"], f"{eval_input.uid}.ipynb")
        with open(outfile, "w") as f:
            f.write(interaction_output["ipynb_str"])

        return outfile


def run_codenav_on_query(
    exp_name: str,
    out_dir: str,
    query: str,
    code_dir: str,
    index_name: str,
    working_dir: str = os.path.join(
        os.path.dirname(ABS_PATH_OF_CODENAV_DIR), "playground"
    ),
    sys_paths: Optional[List[str]] = None,
    repo_description_path: Optional[str] = None,
    es_host: str = DEFAULT_ES_HOST,
    max_steps: int = 20,
):

    prompt_dirs = [PROMPTS_DIR]
    repo_description = "default/repo_description.txt"
    if repo_description_path is not None:
        prompt_dir, repo_description = os.path.split(repo_description_path)
        conflict_path = os.path.join(PROMPTS_DIR, repo_description)
        if os.path.exists(conflict_path):
            raise ValueError(
                f"Prompt conflict detected: {repo_description} already exists in {PROMPTS_DIR}. "
                f"Please rename the {repo_description_path} file to resolve this conflict."
            )
        prompt_dirs.append(prompt_dir)

    episode_kwargs = dict(
        allowed_actions=["done", "code", "search"],
        repo_description=repo_description,
        retrievals_per_keyword=3,
        prototypes_per_keyword=7,
        llm=DEFAULT_OPENAI_MODEL,
        code_dir=code_dir,
        sys_paths=[] if sys_paths is None else sys_paths,
        working_dir=working_dir,
        index_name=index_name,
        host=es_host,
        prompt_dirs=prompt_dirs,
    )
    interaction_kwargs = dict(max_steps=max_steps, verbose=True)
    logging_kwargs = dict(out_dir=out_dir)

    # Run CodeNav on the query
    outfile = CodenavEvaluator.evaluate_input(
        eval_input=EvalInput(uid=exp_name, query=query),
        eval_spec=DefaultEvalSpec(
            episode_kwargs=episode_kwargs,
            interaction_kwargs=interaction_kwargs,
            logging_kwargs=logging_kwargs,
        ),
    )

    print("Output saved to ", outfile)
