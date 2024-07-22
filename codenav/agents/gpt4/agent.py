from typing import Optional, Sequence

import codenav.interaction.messages as msg
from codenav.agents.interaction_formatters import InteractionFormatter
from codenav.agents.llm_chat_agent import LLMChatCodeNavAgent, LlmChatMessage
from codenav.constants import DEFAULT_OPENAI_MODEL, OPENAI_CLIENT
from codenav.utils.llm_utils import create_openai_message


class OpenAICodeNavAgent(LLMChatCodeNavAgent):
    def __init__(
        self,
        prompt: str,
        model: str = DEFAULT_OPENAI_MODEL,
        max_tokens: int = 50000,
        allowed_action_types: Sequence[msg.ACTION_TYPES] = (
            "code",
            "done",
            "search",
            "reset",
        ),
        interaction_formatter: Optional[InteractionFormatter] = None,
    ):
        super().__init__(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            allowed_action_types=allowed_action_types,
            interaction_formatter=interaction_formatter,
        )

    @property
    def client(self):
        return OPENAI_CLIENT

    def create_message_from_text(self, text: str, role: str) -> LlmChatMessage:
        return create_openai_message(text=text, role=role)
