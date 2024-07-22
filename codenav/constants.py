import os
import warnings
from pathlib import Path
from typing import Optional

ABS_PATH_OF_CODENAV_DIR = os.path.abspath(os.path.dirname(Path(__file__)))
PROMPTS_DIR = os.path.join(ABS_PATH_OF_CODENAV_DIR, "prompts")

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-05-13"
DEFAULT_RETRIEVAL_PER_QUERY = 3


def get_env_var(name: str) -> Optional[str]:
    if name in os.environ:
        return os.environ[name]
    else:
        return None


OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
OPENAI_ORG = get_env_var("OPENAI_ORG")
OPENAI_CLIENT = None
try:
    from openai import OpenAI

    if OPENAI_API_KEY is not None and OPENAI_ORG is not None:
        OPENAI_CLIENT = OpenAI(
            api_key=OPENAI_API_KEY,
            organization=OPENAI_ORG,
        )
    else:
        warnings.warn(
            "OpenAI_API_KEY and OPENAI_ORG not set. OpenAI API will not work."
        )
except ImportError:
    warnings.warn("openai package not found. OpenAI API will not work.")


TOGETHER_API_KEY = get_env_var("TOGETHER_API_KEY")
TOGETHER_CLIENT = None
try:
    from together import Together

    if TOGETHER_API_KEY is not None:
        TOGETHER_CLIENT = Together(api_key=TOGETHER_API_KEY)
    else:
        warnings.warn("TOGETHER_API_KEY not set. Together API will not work.")
except ImportError:
    warnings.warn("together package not found. Together API will not work.")


COHERE_API_KEY = get_env_var("COHERE_API_KEY")
COHERE_CLIENT = None
try:
    import cohere

    if COHERE_API_KEY is not None:
        COHERE_CLIENT = cohere.Client(api_key=COHERE_API_KEY)
    else:
        warnings.warn("COHERE_API_KEY not set. Cohere API will not work.")
except ImportError:
    warnings.warn("cohere package not found. Cohere API will not work.")
