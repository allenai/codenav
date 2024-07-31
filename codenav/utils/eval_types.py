from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import attrs

from codenav.interaction.episode import Episode


@attrs.define
class EvalInput:
    uid: Union[str, int]
    query: str
    metadata: Optional[Any] = None


Str2AnyDict = Dict[str, Any]


class EvalSpec(ABC):
    def __init__(
        self,
        episode_kwargs: Str2AnyDict,
        interaction_kwargs: Str2AnyDict,
        logging_kwargs: Str2AnyDict,
    ):
        self.episode_kwargs = episode_kwargs
        self.interaction_kwargs = interaction_kwargs
        self.logging_kwargs = logging_kwargs

    @staticmethod
    @abstractmethod
    def build_episode(
        eval_input: EvalInput,
        episode_kwargs: Optional[Str2AnyDict] = None,
    ) -> Episode:
        pass

    @staticmethod
    @abstractmethod
    def run_interaction(
        episode: Episode,
        interaction_kwargs: Optional[Str2AnyDict] = None,
    ) -> Optional[Str2AnyDict]:
        pass

    @staticmethod
    @abstractmethod
    def log_output(
        interaction_output: Str2AnyDict,
        eval_input: EvalInput,
        logging_kwargs: Optional[Str2AnyDict] = None,
    ) -> Any:
        pass
