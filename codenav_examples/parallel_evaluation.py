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
from codenav.utils.prompt_utils import PromptBuilder


# EvalSpec defines the components for running an evaluation
# EvalSpec is then used by the CodenavEvaluator to run CodeNav on inputs using 1 or more processes
# EvalSpec requires defining 3 methods: build_episode, run_interaction, log_output
class CodenavEvalSpec(EvalSpec):
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

        outfile = os.path.join(logging_kwargs["outdir"], f"{eval_input.uid}.ipynb")
        with open(outfile, "w") as f:
            f.write(interaction_output["ipynb_str"])

        return outfile


def run_parallel_evaluation(
    eval_inputs: List[EvalInput],
    episode_kwargs: Str2AnyDict,
    interaction_kwargs: Str2AnyDict,
    logging_kwargs: Str2AnyDict,
    num_processes: int = 2,
):

    # create an instance of the CodenavEvaluator using the eval spec
    evaluator = CodenavEvaluator(
        eval_spec=CodenavEvalSpec(
            episode_kwargs=episode_kwargs,
            interaction_kwargs=interaction_kwargs,
            logging_kwargs=logging_kwargs,
        )
    )

    # Get outputs from the output queue
    num_inputs = len(eval_inputs)
    for i, output in enumerate(evaluator.evaluate(eval_inputs, n_procs=2)):
        print(
            f"Evaluated {i+1}/{num_inputs} | Input uid: {eval_inputs[i].uid} | Output saved to ",
            output,
        )


if __name__ == "__main__":
    episode_kwargs = dict(
        allowed_actions=["done", "code", "search"],
        repo_description="codenav/repo_description.txt",
        retrievals_per_keyword=3,
        prototypes_per_keyword=7,
        llm=DEFAULT_OPENAI_MODEL,
        code_dir=ABS_PATH_OF_CODENAV_DIR,
        sys_paths=[os.path.dirname(ABS_PATH_OF_CODENAV_DIR)],
        working_dir=os.path.join(
            os.path.dirname(ABS_PATH_OF_CODENAV_DIR), "playground"
        ),
        index_name="codenav",
        host=DEFAULT_ES_HOST,
    )
    interaction_kwargs = dict(max_steps=10, verbose=True)
    logging_kwargs = dict(outdir="/Users/tanmayg/Code/codenav_test/outputs")

    # Define the inputs to evaluate using EvalInput
    # Each EvalInput instance consists of a unique id (uid), a query, and optionally any metadata
    eval_inputs = [
        EvalInput(uid=1, query="Find the DoneEnv and instantiate it"),
        EvalInput(
            uid=2,
            query="Build the prompt template using PromptBuilder and print all the placeholders",
        ),
    ]
    run_parallel_evaluation(
        eval_inputs,
        episode_kwargs,
        interaction_kwargs,
        logging_kwargs,
        num_processes=2,
    )
