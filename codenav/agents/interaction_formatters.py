import abc
from types import MappingProxyType
from typing import Mapping, Literal

from codenav.interaction.messages import (
    CodeNavAction,
    RESPONSE_TYPES,
    Interaction,
    MultiRetrievalResult,
    RetrievalResult,
)
from codenav.retrieval.elasticsearch.create_index import es_doc_to_string

DEFAULT_ACTION_FORMAT_KWARGS = MappingProxyType(
    dict(include_header=True),
)

DEFAULT_RESPONSE_FORMAT_KWARGS = MappingProxyType(
    dict(
        include_code=False,
        display_updated_vars=True,
        include_query=True,
        include_header=True,
    )
)


class InteractionFormatter(abc.ABC):
    def format_action(self, action: CodeNavAction):
        raise NotImplementedError

    def format_response(self, response: RESPONSE_TYPES):
        raise NotImplementedError


class DefaultInteractionFormatter(InteractionFormatter):
    def __init__(
        self,
        action_format_kwargs: Mapping[str, bool] = DEFAULT_ACTION_FORMAT_KWARGS,
        response_format_kwargs: Mapping[str, bool] = DEFAULT_RESPONSE_FORMAT_KWARGS,
    ):
        self.action_format_kwargs = action_format_kwargs
        self.response_format_kwargs = response_format_kwargs

    def format_action(self, action: CodeNavAction):
        return Interaction.format_action(
            action,
            **self.action_format_kwargs,
        )

    def format_response(self, response: RESPONSE_TYPES):
        return Interaction.format_response(
            response,
            **self.response_format_kwargs,
        )


class CustomRetrievalInteractionFormatter(DefaultInteractionFormatter):
    def __init__(
        self, use_summary: Literal["ifshorter", "always", "never", "prototype"]
    ):
        super().__init__()
        self.use_summary = use_summary

    def format_retrieval_result(self, rr: RetrievalResult, include_query=True):
        res_str = ""
        if include_query:
            res_str += f"QUERY:\n{rr.query}\n\n"

        res_str += "CODE BLOCKS:\n"

        if len(rr.es_docs) == 0:
            if rr.failure_reason is not None:
                res_str += f"Failed to retrieve code blocks: {rr.failure_reason}\n"
            else:
                res_str += "No code blocks found.\n"

            return res_str

        for doc in rr.es_docs[: rr.max_expanded]:
            if self.use_summary == "prototype":
                doc = {**doc}
                doc["text"] = doc["prototype"] or doc["text"]

                use_summary = "never"
            else:
                use_summary = self.use_summary

            res_str += f"---\n{es_doc_to_string(doc, use_summary=use_summary)}\n"

        res_str += "---\n"

        unexpanded_docs = rr.es_docs[rr.max_expanded :]
        if len(unexpanded_docs) <= rr.max_expanded:
            res_str += "(All code blocks matching the query were returned.)\n"
        else:
            res_str += (
                f"({len(unexpanded_docs)} additional code blocks not shown."
                f" Search again with the same query to see additional results.)\n\n"
            )

            if rr.max_prototype > 0:
                prototypes_docs = [
                    doc
                    for doc in unexpanded_docs
                    if doc["type"] in {"CLASS", "FUNCTION"}
                ]
                num_prototype_docs_shown = min(len(prototypes_docs), rr.max_prototype)
                res_str += (
                    f"Prototypes for the next {num_prototype_docs_shown} out of"
                    f" {len(prototypes_docs)} classes/functions found in unexpanded results"
                    f" (search again with the same query to see details):\n"
                )
                for doc in prototypes_docs[:num_prototype_docs_shown]:
                    res_str += f"{es_doc_to_string(doc, prototype=True)}\n"

        return res_str

    def format_response(self, response: RESPONSE_TYPES):
        if not isinstance(response, MultiRetrievalResult):
            return super(CustomRetrievalInteractionFormatter, self).format_response(
                response
            )
        else:
            res_str = ""
            for res in response.retrieval_results:
                res_str += self.format_retrieval_result(res, include_query=True)
                res_str += "\n"

            return res_str
