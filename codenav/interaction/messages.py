import abc
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, get_args

import attrs

from codenav.retrieval.elasticsearch.create_index import EsDocument, es_doc_to_string
from codenav.utils.linting_and_type_checking_utils import (
    CodeAnalysisError,
    LintingError,
    TypeCheckingError,
)
from codenav.utils.string_utils import get_tag_content_from_text


class CodeNavMessage:
    @abc.abstractmethod
    def format(self, *args, **kwargs) -> str:
        raise NotImplementedError


@attrs.define
class SystemPrompt(CodeNavMessage):
    content: str

    def format(self) -> str:
        return self.content


ACTION_TYPES = Literal[
    "done", "code", "search", "reset", "code_summary", "request_user_message"
]


@attrs.define
class CodeNavAction(CodeNavMessage):
    thought: Optional[str] = None
    type: Optional[ACTION_TYPES] = None
    content: Optional[str] = None

    @staticmethod
    def get_tag_content_from_text(
        text: str,
        tag: Literal[
            "thought",
            "type",
            "content",
            "reset",
            "code",
        ],
    ) -> Optional[str]:
        return get_tag_content_from_text(text=text, tag=tag)

    @staticmethod
    def from_text(text: str) -> "CodeNavAction":
        thought = CodeNavAction.get_tag_content_from_text(text, "thought")
        type = CodeNavAction.get_tag_content_from_text(text, "type")  # type: ignore
        content = CodeNavAction.get_tag_content_from_text(text, "content")

        assert type is None or type in get_args(
            ACTION_TYPES
        ), f"Invalid action type: {type} (valid types are {get_args(ACTION_TYPES)})"

        return CodeNavAction(
            thought=thought,
            type=type,  # type: ignore
            content=content,
        )

    def to_tagged_text(self) -> str:
        return (
            f"<thought>\n{self.thought}\n</thought>"
            f"\n<type>\n{self.type}\n</type>"
            f"\n<content>\n{self.content}\n</content>"
        )

    def format(self) -> str:
        return self.to_tagged_text()


@attrs.define
class InvalidAction(CodeNavMessage):
    reason: str

    def format(self) -> str:
        return str(self)


@attrs.define
class ExecutionResult(CodeNavMessage):
    code_str: str
    stdout: str
    updated_vars: Optional[Dict[str, Any]] = None
    exec_error: Optional[str] = None
    linting_errors: Optional[List[LintingError]] = None
    type_checking_errors: Optional[List[TypeCheckingError]] = None

    @staticmethod
    def format_vars_with_max_len(vars: Dict[str, Any], max_len: int) -> str:
        """Format local variables with a maximum length per string representation."""

        l = []
        for k, v in vars.items():
            str_v = str(v)
            if len(str_v) > max_len:
                str_v = str_v[:max_len] + "..."
            l.append(f'"{k}": {str_v}')

        return "{" + ", ".join(l) + "}"

    def format(
        self,
        include_code: bool,
        display_updated_vars: bool,
        max_local_var_len: int = 500,
    ) -> str:
        res_str = ""

        if include_code:
            res_str = f"```\n{self.code_str}\n```\n"

        if self.stdout is not None and len(self.stdout) > 0:
            if len(self.stdout) > 2000:
                stdout_start = self.stdout[:1000]
                stdout_end = self.stdout[-1000:]
                msg = "STDOUT was too long. Showing only the start and end separated by ellipsis."
                res_str += f"STDOUT ({msg}):\n{stdout_start}\n\n...\n\n{stdout_end}\n"
            else:
                res_str += f"STDOUT:\n{self.stdout}\n"

        if self.exec_error is not None:
            res_str += f"EXECUTION ERROR:\n{self.exec_error}\n"
        elif self.stdout is None or len(self.stdout) == 0:
            # If there was no error or stdout, want to print something to tell the agent
            # that the code was executed
            res_str += "CODE EXECUTED WITHOUT ERROR, STDOUT WAS EMPTY\n"

        if (
            display_updated_vars
            and self.updated_vars is not None
            and len(self.updated_vars) > 0
        ):
            res_str += (
                f"RELEVANT VARIABLES (only shown if string rep. has changed after code exec):"
                f"\n{ExecutionResult.format_vars_with_max_len(self.updated_vars, max_len=max_local_var_len)}\n"
            )

        analysis_errors: List[CodeAnalysisError] = []

        if self.linting_errors is not None:
            analysis_errors.extend(self.linting_errors)

        if self.type_checking_errors is not None:
            analysis_errors.extend(self.type_checking_errors)

        if len(analysis_errors) > 0:
            res_str += "STATIC ANALYSIS ERRORS:\n"
            for err in analysis_errors:
                res_str += f"{err}\n"

        return res_str


@attrs.define
class RetrievalResult:
    query: str
    es_docs: Sequence[EsDocument]
    failure_reason: Optional[str] = None
    max_expanded: int = 3
    max_prototype: int = 10

    def format(
        self,
        include_query: bool = True,
    ) -> str:
        res_str = ""
        if include_query:
            res_str += f"QUERY:\n{self.query}\n\n"

        res_str += "CODE BLOCKS:\n"

        if len(self.es_docs) == 0:
            if self.failure_reason is not None:
                res_str += f"Failed to retrieve code blocks: {self.failure_reason}\n"
            else:
                res_str += "No code blocks found.\n"

            return res_str

        for doc in self.es_docs[: self.max_expanded]:
            res_str += "---\n{}\n".format(es_doc_to_string(doc, prototype=False))

        res_str += "---\n"

        unexpanded_docs = self.es_docs[self.max_expanded :]
        if len(unexpanded_docs) <= 0:
            res_str += "(All code blocks matching the query were returned.)\n"
        else:
            res_str += (
                f"({len(unexpanded_docs)} additional code blocks not shown."
                f" Search again with the same query to see additional results.)\n\n"
            )

            if self.max_prototype > 0:
                prototypes_docs = [
                    doc
                    for doc in unexpanded_docs
                    if doc["type"] in {"CLASS", "FUNCTION"}
                ]
                num_prototype_docs_shown = min(len(prototypes_docs), self.max_prototype)
                res_str += (
                    f"Prototypes for the next {num_prototype_docs_shown} out of"
                    f" {len(prototypes_docs)} classes/functions found in unexpanded results"
                    f" (search again with the same query to see details):\n"
                )
                for doc in prototypes_docs[:num_prototype_docs_shown]:
                    res_str += "{}\n".format(es_doc_to_string(doc, prototype=True))

        return res_str


@attrs.define
class MultiRetrievalResult(CodeNavMessage):
    retrieval_results: Sequence[RetrievalResult]

    def format(
        self,
        include_query: bool = True,
    ) -> str:
        res_str = ""
        for res in self.retrieval_results:
            res_str += res.format(include_query)
            res_str += "\n"

        return res_str


@attrs.define
class UserMessageToAgent(CodeNavMessage):
    message: str

    def format(self) -> str:
        return self.message


class UserQueryToAgent(UserMessageToAgent):
    pass


RESPONSE_TYPES = Union[
    InvalidAction, MultiRetrievalResult, ExecutionResult, UserMessageToAgent
]


@attrs.define
class Interaction:
    action: Optional[CodeNavAction]
    response: Optional[RESPONSE_TYPES] = None
    hidden: bool = False
    hidden_at_index: bool = -1

    @staticmethod
    def format_action(action: CodeNavAction, include_header=True) -> str:
        if include_header:
            return "ACTION:\n" + action.format()

        return action.format()

    @staticmethod
    def format_response(
        response: RESPONSE_TYPES,
        include_code,
        display_updated_vars,
        include_query,
        include_header,
    ) -> str:
        if isinstance(response, ExecutionResult):
            response_type = "Execution Result"
            response_text = response.format(
                include_code=include_code,
                display_updated_vars=display_updated_vars,
            )
        elif isinstance(response, InvalidAction):
            response_type = "Invalid Action"
            response_text = response.format()
        elif isinstance(response, MultiRetrievalResult):
            response_type = "Retrieval Result"
            response_text = response.format(include_query=include_query)
        elif isinstance(response, UserMessageToAgent):
            response_type = "User Message"
            response_text = response.format()
        else:
            raise NotImplementedError()

        if include_header:
            return f"RESPONSE ({response_type}):\n" + response_text

        return response_text


class EpisodeState:
    def __init__(self, system_prompt: SystemPrompt):
        self.system_prompt = system_prompt
        self.interactions: List[Interaction] = []

    def update(self, interaction: Interaction):
        self.interactions.append(interaction)


def format_reponse(
    response: RESPONSE_TYPES,
    include_code=False,
    display_updated_vars=True,
    include_query=True,
) -> str:
    if isinstance(response, ExecutionResult):
        response_text = response.format(
            include_code=include_code,
            display_updated_vars=display_updated_vars,
        )
    elif isinstance(response, InvalidAction):
        response_text = response.format()
    elif isinstance(response, MultiRetrievalResult):
        response_text = response.format(include_query=include_query)
    elif isinstance(response, UserMessageToAgent):
        response_text = response.format()
    else:
        raise NotImplementedError()

    return "ACTION RESPONSE:\n" + response_text
