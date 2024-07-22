from codenav.agents.gpt4.agent import DEFAULT_OPENAI_MODEL, OpenAICodeNavAgent
from codenav.retrieval.elasticsearch.elasticsearch_constants import RESERVED_CHARACTERS
from codenav.utils.prompt_utils import PromptBuilder

# Prompt builder puts together a prompt template from text files
# The template may contain placeholders for values
prompt_builder = PromptBuilder(repo_description="codenav/repo_description.txt")
prompt_template, placeholder_to_paths = prompt_builder.build_template()

# see placeholders and the file paths they appear in
print("Placeholders in template:\n", placeholder_to_paths)

# provide values for these placeholders
ALLOWED_ACTIONS = ["done", "code", "search"]
placeholder_values = dict(
    AVAILABLE_ACTIONS=ALLOWED_ACTIONS,
    RESERVED_CHARACTERS=RESERVED_CHARACTERS,
    RETRIEVALS_PER_KEYWORD=3,
)
print("Provided values:\n", placeholder_values)

# build prompt using values
# the following is equivalent to prompt_template.format(**placeholder_values)
prompt = prompt_builder.build(placeholder_values)

# create agent using prompt
agent = OpenAICodeNavAgent(
    prompt=prompt,
    model=DEFAULT_OPENAI_MODEL,
    allowed_action_types=ALLOWED_ACTIONS,
)
