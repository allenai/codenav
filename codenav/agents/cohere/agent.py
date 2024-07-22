import copy
import time
from typing import List, Literal, Optional, Sequence

from cohere import ChatMessage, InternalServerError, TooManyRequestsError

import codenav.interaction.messages as msg
from codenav.agents.interaction_formatters import InteractionFormatter
from codenav.agents.llm_chat_agent import LLMChatCodeNavAgent, LlmChatMessage
from codenav.constants import COHERE_CLIENT


class CohereCodeNavAgent(LLMChatCodeNavAgent):
    def __init__(
        self,
        prompt: str,
        model: Literal["command-r", "command-r-plus"] = "command-r",
        max_tokens: int = 50000,
        allowed_action_types: Sequence[msg.ACTION_TYPES] = (
            "code",
            "done",
            "search",
            "reset",
        ),
        prompt_set: str = "default",
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
        return COHERE_CLIENT

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

        output = None
        nretries = 50
        for retry in range(nretries):
            try:
                output = self.client.chat(
                    message=messages[-1]["message"],
                    model=model,
                    chat_history=messages[:-1],
                    prompt_truncation="OFF",
                    temperature=0.0,
                    max_input_tokens=max_tokens,
                    max_tokens=3000,
                )
                break
            except (TooManyRequestsError, InternalServerError):
                pass

            if retry >= nretries - 1:
                raise RuntimeError(f"Hit max retries ({nretries})")

            time.sleep(5)

        self._all_queries_and_responses.append(
            {
                "input": copy.deepcopy(messages),
                "output": output.text,
                "input_tokens": output.meta.billed_units.input_tokens,
                "output_tokens": output.meta.billed_units.output_tokens,
            }
        )
        return output.text

    def create_message_from_text(self, text: str, role: str) -> ChatMessage:
        role = {
            "assistant": "CHATBOT",
            "user": "USER",
            "system": "SYSTEM",
        }[role]

        return dict(role=role, message=text)
