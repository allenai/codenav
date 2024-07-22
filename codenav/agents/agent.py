import abc
from typing import Optional, Sequence, List, Dict, Any

import codenav.interaction.messages as msg
from codenav.agents.interaction_formatters import InteractionFormatter


class CodeNavAgent(abc.ABC):
    max_tokens: int
    interaction_formatter: InteractionFormatter
    model: str

    def __init__(self, allowed_action_types: Sequence[msg.ACTION_TYPES]):
        self.allowed_action_types = allowed_action_types
        # self.reset()

    def reset(self):
        """Reset episode state. Must be called before starting a new episode."""
        self.episode_state = self.init_episode_state()

    @abc.abstractmethod
    def init_episode_state(self) -> msg.EpisodeState:
        """Build system prompt and initialize the episode state with it"""
        raise NotImplementedError

    def update_state(self, interaction: msg.Interaction):
        self.episode_state.update(interaction)

    @property
    def system_prompt_str(self) -> str:
        return self.episode_state.system_prompt.content

    @property
    def user_query_prompt_str(self) -> str:
        queries = [
            i.response
            for i in self.episode_state.interactions
            if isinstance(i.response, msg.UserQueryToAgent)
        ]
        assert len(queries) == 1
        return queries[0].message

    @abc.abstractmethod
    def get_action(self) -> msg.CodeNavAction:
        raise NotImplementedError

    @abc.abstractmethod
    def summarize_episode_state_for_restart(
        self, episode_state: msg.EpisodeState, max_tokens: Optional[int]
    ) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_queries_and_responses(self) -> List[Dict[str, Any]]:
        raise NotImplementedError
