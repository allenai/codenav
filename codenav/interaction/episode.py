from typing import Any, Dict, Optional, Tuple

import pandas as pd

import codenav.interaction.messages as msg
from codenav.agents.agent import CodeNavAgent
from codenav.agents.interaction_formatters import (
    DefaultInteractionFormatter,
    InteractionFormatter,
)
from codenav.environments.abstractions import CodeNavEnv
from codenav.environments.code_env import PythonCodeEnv
from codenav.environments.retrieval_env import RetrievalEnv
from codenav.prompts.query_prompt import create_user_query_message
from codenav.prompts.restart_prompt import PICKUP_PROMPT


class Episode:
    def __init__(
        self,
        agent: CodeNavAgent,
        action_type_to_env: Dict[msg.ACTION_TYPES, CodeNavEnv],
        user_query_str: str,
    ):
        self.agent = agent
        self.action_type_to_env = action_type_to_env

        self.user_query_str = user_query_str
        self.agent.reset()
        # self.agent.reset(action_type_to_env=self.action_type_to_env)

        assert all(k in action_type_to_env for k in ["done", "code"])

    @property
    def code_env(self) -> PythonCodeEnv:
        code_envs = [
            env
            for env in self.action_type_to_env.values()
            if isinstance(env, PythonCodeEnv)
        ]
        assert len(code_envs) == 1
        return code_envs[0]

    def check_action_validity(self, action: msg.CodeNavAction) -> Tuple[bool, str]:
        is_valid = True
        error_msg = ""

        if action.type == "reset":
            return True, ""

        if action.thought is None:
            is_valid = False
            error_msg += "Action should always contain thought.\n"

        if action.type not in self.action_type_to_env:
            is_valid = False
            error_msg += f"Action type {action.type} is not supported.\n"
            return is_valid, error_msg

        assert action.type is not None
        env = self.action_type_to_env[action.type]

        content_valid, error_msg_content = env.check_action_validity(action)

        is_valid = is_valid and content_valid
        error_msg += error_msg_content

        return is_valid, error_msg

    def step(self) -> msg.Interaction:
        if len(self.agent.episode_state.interactions) == 0:
            assert self.code_env.code_dir is not None

            # Start of episode, add user query
            self.agent.update_state(
                msg.Interaction(
                    action=None,
                    response=create_user_query_message(
                        user_query_str=self.user_query_str,
                        code_dir=self.code_env.code_dir,
                        working_dir=self.code_env.working_dir,
                        added_paths=self.code_env.sys_paths,
                    ),
                )
            )

        action = self.agent.get_action()

        action_is_valid, action_error_msg = self.check_action_validity(action)

        response: Optional[msg.RESPONSE_TYPES]
        if action_is_valid and action.type == "reset":
            response = msg.UserMessageToAgent(message=PICKUP_PROMPT)
            for env in self.action_type_to_env.values():
                if isinstance(env, RetrievalEnv):
                    env.reset()

        elif action_is_valid:
            assert action.type is not None
            try:
                response = self.action_type_to_env[action.type].step(action)
            except KeyboardInterrupt:
                print(
                    f"Keyboard interrupt occurred while attempting to execute code:{{\n{action.content}\n}}\n"
                    f"String of notebook before interrupt: {self.to_notebook(cur_dir=self.code_env.working_dir)}\n",
                    flush=True,
                )
                raise
        else:
            response = msg.InvalidAction(reason=action_error_msg)

        interaction = msg.Interaction(action=action, response=response)
        self.agent.update_state(interaction=interaction)

        return interaction

    def step_until_max_steps_or_success(self, max_steps: int, verbose: bool = True):
        for i in range(max_steps):
            interaction = self.step()
            if verbose:
                print("*" * 80)
                print(f"Step {i+1}")
                print("*" * 80)
                print("")
                print(
                    Episode.format_interaction(
                        interaction, self.agent.interaction_formatter
                    )
                )
            if (
                interaction.action is not None
                and interaction.action.type == "done"
                and not isinstance(interaction.response, msg.InvalidAction)
            ):
                break

    @staticmethod
    def get_record(
        interaction: msg.Interaction, formatter: InteractionFormatter
    ) -> dict[str, Any]:
        if formatter is None:
            formatter = DefaultInteractionFormatter()

        if interaction.response is None:
            response_text = None
        else:
            response_text = formatter.format_response(
                interaction.response,
            )

        action = (
            msg.CodeNavAction() if interaction.action is None else interaction.action
        )
        return {
            "action/thought": action.thought,
            "action/type": str(action.type),
            "action/content": action.content,
            "response": response_text,
            "hidden": interaction.hidden,
        }

    def tabulate_interactions(self) -> pd.DataFrame:
        records = []
        for interaction in self.agent.episode_state.interactions:
            records.append(
                Episode.get_record(interaction, self.agent.interaction_formatter)
            )

        return pd.DataFrame.from_records(records)

    def tabulate_exec_trace(self) -> pd.DataFrame:
        return self.code_env.tabulate()

    def tabulate_prompts(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            dict(
                prompt_type=["QUERY", "SYSTEM_PROMPT"],
                prompt_content=[
                    self.agent.user_query_prompt_str,
                    self.agent.system_prompt_str,
                ],
            )
        )

    def to_notebook(self, cur_dir: str) -> str:
        import nbformat
        from nbformat import v4 as nbf

        nb = nbf.new_notebook()
        for i, row in self.tabulate_interactions().iterrows():
            thought = row["action/thought"]
            action_type = row["action/type"]
            content = row["action/content"]
            response = row["response"]
            thought, action_type, content, response = (
                x if x is not None else "None"
                for x in (thought, action_type, content, response)
            )

            if i == 0:
                nb.cells.append(
                    nbf.new_markdown_cell(
                        "\n\n".join(
                            [
                                "# Instruction to CodeNav",
                                self.agent.user_query_prompt_str,
                                "# Interactions",
                            ]
                        )
                    )
                )
                continue

            nb.cells.append(
                nbf.new_markdown_cell(
                    "\n\n".join(
                        [
                            f"## Step {i}: {action_type}",
                            thought,
                        ]
                    )
                )
            )

            if action_type == "done":
                output = nbf.new_output(
                    output_type="stream",
                    text="Ending episode since the agent has issued a 'done' action.",
                )
            elif action_type == "code":
                output = nbf.new_output(
                    output_type="execute_result",
                    data={"text/plain": response},
                )
            else:
                output = nbf.new_output(
                    output_type="stream",
                    text=response,
                )

            nb.cells.append(nbf.new_code_cell(content, outputs=[output]))
            nb.cells.append(nbf.new_markdown_cell("---"))

        return nbformat.writes(nb)

    @staticmethod
    def format_interaction(
        interaction: msg.Interaction, interaction_formatter: InteractionFormatter
    ) -> str:
        record = Episode.get_record(interaction, formatter=interaction_formatter)
        return Episode.format_record(record)

    @staticmethod
    def format_record(record: dict[str, Any]) -> str:
        inter_str = "------Action------"

        inter_str += f"\nTHOUGHT:\n{record['action/thought']}"

        inter_str += f"\nACTION TYPE:\n{record.get('action/type')}"

        inter_str += f"\nACTION CONTENT:\n{record.get('action/content')}"

        inter_str += "\n\n-----Response-----"
        inter_str += f"\n{record['response']}"
        return inter_str
