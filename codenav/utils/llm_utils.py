import base64
import json
import math
import time
import traceback
from typing import List, Literal, Optional, Sequence, Union, cast, Any, Dict

import numpy as np
import tiktoken
import tqdm
from PIL import Image
from openai import InternalServerError, OpenAI, RateLimitError
from openai.resources import Chat
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS = {
    # Together models, input tokens cost same as output
    "mistralai/Mistral-7B-Instruct-v0.2": 0.2,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.6,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 1.2,
    "Qwen/Qwen1.5-72B-Chat": 1.0,
    "Qwen/Qwen1.5-110B-Chat": 1.8,
    "meta-llama/Llama-3-70b-chat-hf": 0.9,
    # OpenAI models
    "gpt-3.5-turbo-0301": 1.5,
    "gpt-3.5-turbo-0125": 1.5,
    "gpt-4-1106-preview": 10.0,
    "gpt-4o-2024-05-13": 5.0,
    # Cohere
    "command-r": 0.5,
    "command-r-plus": 3.0,
}

MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS = {
    # Together models, input tokens cost same as output
    "mistralai/Mistral-7B-Instruct-v0.2": 0.2,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.6,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 1.2,
    "Qwen/Qwen1.5-72B-Chat": 1.0,
    "Qwen/Qwen1.5-110B-Chat": 1.8,
    "meta-llama/Llama-3-70b-chat-hf": 0.9,
    # OpenAI models
    "gpt-3.5-turbo-0301": 2.0,
    "gpt-3.5-turbo-0125": 2.0,
    "gpt-4o-2024-05-13": 15.0,
    "gpt-4-1106-preview": 30.0,
    # Cohere
    "command-r": 1.5,
    "command-r-plus": 15.0,
}


class MaxTokensExceededError(Exception):
    pass


def num_tokens_from_messages(
    messages,
    skip_images: bool,
    model="gpt-3.5-turbo-0301",
):
    """Returns the number of tokens used by a list of messages."""
    assert skip_images, "skip_images=False is not presently supported"

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model in [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0125",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
    ]:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                if key == "content":
                    if isinstance(value, str):
                        num_tokens += len(encoding.encode(value, disallowed_special=[]))
                    elif isinstance(value, List):
                        for piece in value:
                            if piece["type"] == "text":
                                num_tokens += len(
                                    encoding.encode(
                                        piece["text"], disallowed_special=[]
                                    )
                                )
                            else:
                                assert skip_images
                    else:
                        raise NotImplementedError
                elif key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
                else:
                    num_tokens += len(encoding.encode(value, disallowed_special=[]))
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def image_height_width_from_path(image_path: str):
    with Image.open(image_path) as img:
        # Load only image metadata (not pixel data)
        img.load()

        # Get dimensions
        width, height = img.size

        return height, width


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=[]))
    return num_tokens


def compute_token_count_for_image(image: Union[str, np.ndarray]):
    # From https://platform.openai.com/docs/guides/vision
    if isinstance(image, str):
        h, w = image_height_width_from_path(image)
    else:
        h, w, _ = image.shape

    # First rescaled to be within 2048x2048
    scale = 2048 / max([2048, h, w])
    h = h * scale
    w = w * scale

    # Then rescaled so shortest edge is 768
    h, w = 768 * h / min(h, w), 768 * w / min(h, w)

    return math.ceil(h / 512) * math.ceil(w / 512) * 170 + 85


def partition_sequence(seq: Sequence, parts: int) -> List:
    assert 0 < parts, f"parts [{parts}] must be greater > 0"
    assert parts <= len(seq), f"parts [{parts}] > len(seq) [{len(seq)}]"
    n = len(seq)

    quotient = n // parts
    remainder = n % parts
    counts = [quotient + (i < remainder) for i in range(parts)]
    inds = np.cumsum([0] + counts)
    return [seq[ind0:ind1] for ind0, ind1 in zip(inds[:-1], inds[1:])]


def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy objects.

    Based off the stackoverflow answer by Jie Yang here: https://stackoverflow.com/a/57915246.
    The license for this code is [BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
    """

    def default(self, obj):
        if isinstance(obj, np.void):
            return None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def compute_llm_cost(input_tokens: int, output_tokens: int, model: str):
    assert (
        model in MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS
        and model in MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS
    ), f"model [{model}] must be in both MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS and MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS"

    input_token_cost_per_1m = MODEL_STR_TO_PRICE_PER_1M_INPUT_TOKENS[model]
    output_token_cost_per_1m = MODEL_STR_TO_PRICE_PER_1M_OUTPUT_TOKENS[model]

    return (
        input_tokens * input_token_cost_per_1m
        + output_tokens * output_token_cost_per_1m
    ) / 1e6


def compute_cost_for_queries_and_responses(
    queries_and_responses: Sequence[Dict[str, Union[str, List[Dict[str, Any]]]]],
    model: str,
    encoding_name: Optional[str] = "cl100k_base",
):
    input_tokens = 0
    output_tokens = 0
    for query_and_response in queries_and_responses:
        if "input_tokens" in query_and_response:
            input_tokens += query_and_response["input_tokens"]
        else:
            input_tokens += num_tokens_from_messages(
                query_and_response["input"], model=model, skip_images=True
            )

        if "output_tokens" in query_and_response:
            output_tokens += query_and_response["output_tokens"]
        else:
            output_str = query_and_response["output"]
            output_tokens += len(
                tiktoken.get_encoding(encoding_name).encode(
                    output_str, disallowed_special=[]
                )
            )

    return (
        compute_llm_cost(input_tokens, output_tokens, model=model),
        input_tokens,
        output_tokens,
    )


def create_openai_message(
    text: str, role: Literal["user", "system", "assistant"] = "user"
) -> Union[
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
]:
    return {  # type: ignore
        "role": role,
        "content": [
            {
                "type": "text",
                "text": text,
            },
        ],
    }


def query_gpt(
    messages: Sequence[ChatCompletionMessageParam],
    model: str,
    client: OpenAI,
    pbar: Optional[tqdm.tqdm] = None,
    sec_wait_between_retries: float = 10,
    max_tokens: int = 3000,
    return_input_output_tokens: bool = False,
) -> Optional[Union[str, Dict[str, Union[str, int]]]]:
    """Query the OpenAI API with the given messages."""
    num_tokens = num_tokens_from_messages(messages, skip_images=True)
    if num_tokens > max_tokens:
        raise MaxTokensExceededError(
            f"num_tokens [{num_tokens}] > max_tokens [{max_tokens}]"
        )

    if pbar:
        pbar.write(f"Num tokens: {num_tokens}")

    response = None
    for retry in range(10):
        try:
            response = cast(Chat, client.chat).completions.create(
                model=model,
                messages=cast(List[ChatCompletionMessageParam], messages),
                max_tokens=3000,  # Max number of output tokens
                temperature=0.0,
            )
            break
        except RateLimitError:
            if pbar:
                pbar.write(
                    f"Rate limit error, waiting {sec_wait_between_retries} seconds..."
                )
        except InternalServerError:
            if pbar:
                pbar.write(
                    f"Internal server error, waiting {sec_wait_between_retries} seconds..."
                )
        except:
            m = traceback.format_exc().lower()
            if "ratelimit" in m or "rate limit" in m:
                if pbar:
                    pbar.write(
                        f"Rate limit error, waiting {sec_wait_between_retries} seconds..."
                    )
            else:
                raise

        if response is not None:
            break

        if retry >= 9:
            if pbar:
                pbar.write(f"Hit max retries, raising exception.")
            raise RuntimeError("Hit max retries")

        if pbar:
            pbar.write(
                f"Retry {retry} failed, sleeping for {sec_wait_between_retries} seconds"
            )

        time.sleep(sec_wait_between_retries)

    if response is None:
        return None

    output = response.choices[0].message.content

    if return_input_output_tokens:
        return {
            "output": output,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    else:
        return output
