import abc
from typing import Tuple, Optional, Union

from codenav.interaction.messages import (
    CodeNavAction,
    InvalidAction,
    MultiRetrievalResult,
    ExecutionResult,
    UserMessageToAgent,
)


class CodeNavEnv(abc.ABC):
    @abc.abstractmethod
    def check_action_validity(self, action: CodeNavAction) -> Tuple[bool, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self, action: CodeNavAction
    ) -> Optional[
        Union[InvalidAction, MultiRetrievalResult, ExecutionResult, UserMessageToAgent]
    ]:
        raise NotImplementedError
