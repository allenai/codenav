import copy
import traceback
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import codenav.interaction.messages as msg
from codenav.agents.agent import CodeNavAgent
from codenav.agents.interaction_formatters import (
    DefaultInteractionFormatter,
    InteractionFormatter,
)
from codenav.constants import (
    TOGETHER_CLIENT,
)
from codenav.prompts.restart_prompt import RESTART_PROMPT
from codenav.utils.llm_utils import MaxTokensExceededError, query_gpt

LlmChatMessage = Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
]


class LLMChatCodeNavAgent(CodeNavAgent):
    def __init__(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 50000,
        allowed_action_types: Sequence[msg.ACTION_TYPES] = (
            "code",
            "done",
            "search",
            "reset",
        ),
        interaction_formatter: Optional[InteractionFormatter] = None,
    ):
        super().__init__(allowed_action_types=allowed_action_types)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens

        if interaction_formatter is None:
            self.interaction_formatter = DefaultInteractionFormatter()
        else:
            self.interaction_formatter = interaction_formatter

        self._all_queries_and_responses: List[Dict] = []

    @property
    def client(self):
        return TOGETHER_CLIENT

    def query_llm(
        self,
        messages: List[LlmChatMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if model is None:
            model = self.model

        if max_tokens is None:
            max_tokens = self.max_tokens

        response_dict = query_gpt(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            client=self.client,
            return_input_output_tokens=True,
        )
        self._all_queries_and_responses.append(
            {"input": copy.deepcopy(messages), **response_dict}
        )
        return response_dict["output"]

    @property
    def all_queries_and_responses(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._all_queries_and_responses)

    def init_episode_state(self) -> msg.EpisodeState:
        return msg.EpisodeState(system_prompt=msg.SystemPrompt(content=self.prompt))

    def create_message_from_text(self, text: str, role: str) -> LlmChatMessage:
        return cast(LlmChatMessage, {"role": role, "content": text})

    def build_chat_context(
        self,
        episode_state: msg.EpisodeState,
    ) -> List[LlmChatMessage]:
        chat_messages: List[LlmChatMessage] = [
            self.create_message_from_text(
                text=episode_state.system_prompt.content, role="system"
            ),
        ]
        for interaction in episode_state.interactions:
            if interaction.hidden:
                continue

            if interaction.action is not None:
                chat_messages.append(
                    self.create_message_from_text(
                        text=self.interaction_formatter.format_action(
                            interaction.action
                        ),
                        role="assistant",
                    )
                )

            if interaction.response is not None:
                chat_messages.append(
                    self.create_message_from_text(
                        text=self.interaction_formatter.format_response(
                            interaction.response,
                        ),
                        role="user",
                    )
                )

        return chat_messages

    def summarize_episode_state_for_restart(
        self, episode_state: msg.EpisodeState, max_tokens: Optional[int]
    ) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens

        chat_messages = self.build_chat_context(episode_state)

        current_env_var_names = {
            var_name
            for i in episode_state.interactions
            if isinstance(i.response, msg.ExecutionResult)
            for var_name in (i.response.updated_vars or {}).keys()
        }

        chat_messages.append(
            self.create_message_from_text(
                text=RESTART_PROMPT.format(
                    current_env_var_names=", ".join(sorted(list(current_env_var_names)))
                ),
                role="user",
            )
        )

        return self.query_llm(messages=chat_messages, max_tokens=max_tokens)

    def get_action(self) -> msg.CodeNavAction:
        chat_messages = self.build_chat_context(self.episode_state)
        try:
            output = self.query_llm(messages=chat_messages)
            return msg.CodeNavAction.from_text(output)
        except MaxTokensExceededError:
            if "reset" in self.allowed_action_types:
                summary = self.summarize_episode_state_for_restart(
                    episode_state=self.episode_state, max_tokens=2 * self.max_tokens
                )
                for interaction in self.episode_state.interactions:
                    if not isinstance(interaction.response, msg.UserQueryToAgent):
                        if not interaction.hidden:
                            interaction.hidden_at_index = len(
                                self.episode_state.interactions
                            )
                        interaction.hidden = True
                action = msg.CodeNavAction.from_text(summary)
                action.type = "reset"
                return action

            return msg.CodeNavAction(
                thought=f"Max tokens exceeded:\n{traceback.format_exc()}",
                type="done",
                content="False",
            )
