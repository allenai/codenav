import os

from codenav.agents.gpt4.agent import OpenAICodeNavAgent
from codenav.constants import ABS_PATH_OF_CODENAV_DIR, DEFAULT_OPENAI_MODEL
from codenav.environments.code_env import PythonCodeEnv
from codenav.environments.done_env import DoneEnv
from codenav.environments.retrieval_env import EsCodeRetriever, RetrievalEnv
from codenav.interaction.episode import Episode
from codenav.retrieval.elasticsearch.elasticsearch_constants import RESERVED_CHARACTERS
from codenav.retrieval.elasticsearch.index_codebase import DEFAULT_ES_HOST
from codenav.utils.prompt_utils import PromptBuilder

ALLOWED_ACTIONS = ["done", "code", "search"]
CODE_DIR = ABS_PATH_OF_CODENAV_DIR
PARENT_DIR = os.path.dirname(CODE_DIR)

# create prompt
prompt_builder = PromptBuilder(repo_description="codenav/repo_description.txt")
prompt = prompt_builder.build(
    dict(
        AVAILABLE_ACTIONS=ALLOWED_ACTIONS,
        RESERVED_CHARACTERS=RESERVED_CHARACTERS,
        RETRIEVALS_PER_KEYWORD=3,
    )
)

# create environments
code_env = PythonCodeEnv(
    code_dir=CODE_DIR,
    sys_paths=[PARENT_DIR],
    working_dir=os.path.join(PARENT_DIR, "playground"),
)

retrieval_env = RetrievalEnv(
    code_retriever=EsCodeRetriever(
        index_name="codenav",
        host=DEFAULT_ES_HOST,
    ),
    expansions_per_query=3,
    prototypes_per_query=7,
)

done_env = DoneEnv()


# create agent using prompt
agent = OpenAICodeNavAgent(
    prompt=prompt,
    model=DEFAULT_OPENAI_MODEL,
    allowed_action_types=ALLOWED_ACTIONS,
)

# create environments:
episode = Episode(
    agent,
    action_type_to_env=dict(
        code=code_env,
        search=retrieval_env,
        done=done_env,
    ),
    user_query_str="Find the DoneEnv and instantiate it",
)

episode.step_until_max_steps_or_success(max_steps=5)
