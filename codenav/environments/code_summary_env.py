from typing import Optional, Tuple

from codenav.environments.abstractions import CodeNavEnv
from codenav.interaction.messages import CodeNavAction, UserMessageToAgent


class CodeSummaryEnv(CodeNavEnv):
    def __init__(self) -> None:
        self.summary: Optional[str] = None

    def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
        if action.content is None or action.content.strip() == "":
            return False, "No summary found in the action content."

        return True, ""

    def step(self, action: CodeNavAction) -> UserMessageToAgent:
        assert action.content is not None
        self.summary = action.content.strip()
        return UserMessageToAgent(message="Summary received and stored.")
