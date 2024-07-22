from typing import Tuple

from codenav.environments.abstractions import CodeNavEnv
from codenav.interaction.messages import CodeNavAction


class DoneEnv(CodeNavEnv):
    def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
        assert action.content is not None

        if action.content.strip().lower() in ["true", "false"]:
            return True, ""
        else:
            return (
                False,
                "When executing the done action, the content must be either 'True' or 'False'",
            )

    def step(self, action: CodeNavAction) -> None:
        return None
