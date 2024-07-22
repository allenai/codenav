import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import attrs

from codenav.constants import PROMPTS_DIR


def extract_placeholders(s):
    pattern = r"(?<!\\)\{([^{}]+)\}(?!\})"

    # Find all matches
    matches = re.findall(pattern, s)
    return matches


@attrs.define
class ActionPrompts:
    preamble: str = "default/action__preamble.txt"
    search: str = "default/action__es_search.txt"
    code: str = "default/action__code.txt"
    done: str = "default/action__done.txt"
    guidelines: str = "default/action__guidelines.txt"


@attrs.define
class ResponsePrompts:
    preamble: str = "default/response__preamble.txt"
    search: str = "default/response__es_search.txt"
    code: str = "default/response__code.txt"
    done: str = "default/response__done.txt"


@attrs.define
class PromptBuilder:
    overview: str = "default/overview.txt"
    workflow: str = "default/workflow.txt"
    action_prompts: ActionPrompts = ActionPrompts()
    response_prompts: ResponsePrompts = ResponsePrompts()
    repo_description: str = "default/repo_description.txt"
    prompt_dirs: Sequence[str] = [PROMPTS_DIR]
    actions_to_enable = ["done", "code", "search"]
    placeholder_to_paths: Dict[str, List[str]] = defaultdict(list)

    def get_prompt(self, rel_path: str) -> str:
        found_path = None
        for prompt_dir in self.prompt_dirs:
            prompt_path = os.path.join(prompt_dir, rel_path)
            if os.path.exists(prompt_path):
                found_path = prompt_path
                break

        if found_path is None:
            raise FileNotFoundError(
                f"Prompt file with relative path: '{rel_path}' not found "
                f"in any of the prompt directories: {self.prompt_dirs}"
            )

        with open(found_path, "r") as f:
            prompt_str = f.read()
            placeholders = extract_placeholders(prompt_str)
            for pl in placeholders:
                self.placeholder_to_paths[pl].append(found_path)

            return prompt_str

    def get_action_prompts(self) -> Sequence[str]:
        return [
            self.get_prompt(getattr(self.action_prompts, action))
            for action in self.actions_to_enable
        ]

    def get_response_prompts(self) -> Sequence[str]:
        return [
            self.get_prompt(getattr(self.response_prompts, action))
            for action in self.actions_to_enable
        ]

    def build_template(self) -> Tuple[str, Dict[str, List[str]]]:
        self.placeholder_to_paths = defaultdict(list)
        prompts = [
            self.get_prompt(self.overview),
            self.get_prompt(self.workflow),
            self.get_prompt(self.action_prompts.preamble),
            *self.get_action_prompts(),
            self.get_prompt(self.action_prompts.guidelines),
            self.get_prompt(self.response_prompts.preamble),
            *self.get_response_prompts(),
            self.get_prompt(self.repo_description),
        ]

        return "\n\n".join(prompts), dict(self.placeholder_to_paths)

    def build(self, placeholder_values: Dict[str, Any]) -> str:
        return self.build_template()[0].format(**placeholder_values)
