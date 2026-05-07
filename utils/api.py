import functools
import warnings

import backoff
import openai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses import Response
from tqdm.contrib.concurrent import thread_map
from transformers import AutoTokenizer

# osi
OPENAI_API_KEY = "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"
GEMINI_API_KEY = None


def get_default_openai_api_key() -> str:
    return OPENAI_API_KEY


def get_default_gemini_api_key() -> str:
    raise ValueError("GEMINI_API_KEY is not set")


@functools.lru_cache
def get_openai_client(api_key: str = None) -> OpenAI:
    if api_key is None:
        api_key = get_default_openai_api_key()
    return OpenAI(api_key=api_key)


@functools.lru_cache
def get_gemini_client(api_key: str = None) -> OpenAI:
    if api_key is None:
        api_key = get_default_gemini_api_key()
    return OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


@functools.lru_cache
def get_local_client(port: int = 32001) -> OpenAI:
    return OpenAI(api_key="empty", base_url=f"http://localhost:{port}/v1")


API_COSTS = {
    "gpt-5": {
        "input": 1.25 / 1_000_000,
        "output": 10 / 1_000_000,
    },
    "gpt-5-nano": {
        "input": 0.05 / 1_000_000,
        "output": 0.4 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.25 / 1_000_000,
        "output": 2 / 1_000_000,
    },
    # Gemini prices assume <200K context. Actual cost is higher for longer contexts.
    "gemini-2.5-pro": {
        "input": 1.25 / 1_000_000,
        "output": 10 / 1_000_000,
    },
    "gemini-2.5-flash": {
        "input": 0.30 / 1_000_000,
        "output": 2.5 / 1_000_000,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10 / 1_000_000,
        "output": 0.40 / 1_000_000,
    },
}

def get_model_api_type(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("gpt-"):
        return "openai"
    if model_lower.startswith("gemini-"):
        return "gemini"
    if model_lower.startswith("qwen/"):
        return "vllm"
    raise NotImplementedError(f"Only gpt-, gemini-, and qwen/ models are supported, got: {model}")


def get_client(model: str, port: int = None) -> OpenAI:
    model_api_type = get_model_api_type(model)
    if model_api_type == "openai":
        return get_openai_client()
    if model_api_type == "gemini":
        return get_gemini_client()
    if port is None:
        raise ValueError("Port is required for local vLLM models")
    return get_local_client(port)


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=60, max_tries=6)
def responses_with_backoff(model=None, port=None, client=None, **kwargs) -> Response:
    if client is None:
        client = get_client(model, port)
    model_api_type = get_model_api_type(model)

    if model_api_type == "openai":
        return client.responses.create(model=model, **kwargs)
    else:
        # Use old chat completions API for other models
        ## Format input as a list of messages
        input = kwargs.pop("input")
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            raise ValueError(f"Input must be a string for Gemini models")

        ## Adapt reasoning effort parameter for Gemini models
        if model_api_type == "gemini":
            reasoning = kwargs.pop("reasoning", None)
            if reasoning is not None and reasoning.get("effort", None) is not None:
                kwargs["reasoning_effort"] = reasoning["effort"]

        return client.chat.completions.create(model=model, messages=messages, **kwargs)


def get_response_cost(response: Response | ChatCompletion) -> float:
    keys = list(API_COSTS.keys())
    keys.sort(key=len, reverse=True)  # longer keys first since some keys are prefixes of others
    for key in keys:
        if response.model.startswith(key):
            costs = API_COSTS[key]
            break
    else:
        warnings.warn(
            f"Model {response.model} not found in API_COSTS. Assuming cost is 0.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    if isinstance(response, Response):
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
    else:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    total_cost = input_tokens * costs["input"] + output_tokens * costs["output"]

    if isinstance(response, Response):
        if response.service_tier == "flex":
            total_cost *= 0.5
        elif response.service_tier == "priority":
            total_cost *= 2

    return total_cost


def get_responses(
    model: str,
    inputs: list[str],
    reasoning_effort: str | None = None,
    port: int = None,
    max_workers: int = 256,
    **response_kwargs,
) -> tuple[list[str | None], list[float], list[Response | ChatCompletion | None]]:
    """
    Pass raw prompt strings as `inputs` (no manual chat/message templating).

    Usage examples:

    # GPT-5 (responses API)
    response_texts, costs_usd, responses = get_responses(
        model="gpt-5",
        inputs=inputs,
        reasoning_effort="medium",
    )

    # Local (chat completions API) (calling vLLM server)
    response_kwargs = get_qwen3_kwargs(max_tokens=8192, thinking=True)
    response_texts, costs_usd, responses = get_responses(
        model="Qwen/Qwen3-1.7B",
        inputs=inputs,
        port=32001,
        **response_kwargs,
    )
    """

    model_api_type = get_model_api_type(model)

    def get_response(input: str) -> Response | ChatCompletion | None:
        request_kwargs = dict(response_kwargs)
        if reasoning_effort is not None:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        try:
            return responses_with_backoff(
                model=model,
                port=port,
                input=input,
                **request_kwargs,
            )
        except Exception as e:
            print(f"Error getting response for input: {input}")
            print(f"Error: {e}")
            print("Setting response to None")
            return None

    responses = thread_map(
        get_response, inputs, max_workers=max_workers, desc=f"Getting responses from {model} via API"
    )
    costs_usd = [get_response_cost(response) if response is not None else 0.0 for response in responses]

    if model_api_type == "openai":
        response_texts = [response.output_text if response is not None else None for response in responses]
    else:
        response_texts = [
            (response.choices[0].message.content if response is not None else None) for response in responses
        ]

    return response_texts, costs_usd, responses


def get_qwen3_kwargs(max_tokens: int = 8192, thinking: bool = False) -> dict:
    if thinking:
        return {
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
    else:
        return {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }


def get_qwen3_5_kwargs(max_tokens: int = 8192, thinking: bool = False) -> dict:
    if thinking:
        return {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        }
    else:
        return {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
