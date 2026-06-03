import os
import time
import openai
# from sklearn import logger
from loguru import logger
import tiktoken
import httpx
from dotenv import load_dotenv
import sys
import json
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
load_dotenv()

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
)


def run_llm(
    prompt,
    temperature=0.7,
    max_tokens=256,
    opeani_api_keys=None,
    engine="gpt-3.5-turbo-0613",
    stop="\n",
    stream=False,
    n=1,
):
    client = openai.OpenAI(
        # base_url=os.environ['base_url'],
        # api_key=os.environ['opeani_api_keys'],
        # http_client=httpx.Client(proxies=os.environ['custom_proxy']) if 'custom_proxy' in os.environ else None,
    )
    # print("prompt:", prompt)
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers complex questions."}
    ]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)

    if stop and type(stop) is str:
        stop = [stop]

    f = 0
    while f == 0:
        try:
            if len(encoding.encode(prompt)) >= 4096:
                raise RuntimeError("maximum context length of prompt")
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
                n=n,
            )
            results = [response.choices[i].message.content for i in range(n)]

            if stop:
                stop = stop[0]
                for i, result in enumerate(results): 
                    if stop in result:
                        result = result.split(stop)[0]
                    results[i] = result
                    
            f = 1
        except Exception as e:
            if "maximum context length" in str(e):
                logger.error(f"{e}")
                return None
            logger.error(f"{e}, openai error, retry")
            time.sleep(5)
    if n == 1:
        return results[0]
    else:
        return results


def run_openai_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: str = "gpt-4o-mini",
    retry: int = 3,
    json_output: bool = False,
):
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    llm_response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    response_content = llm_response.choices[0].message.content.strip()
    if json_output:
        return json.loads(response_content)
    return response_content


def run_llm_v2(
    system_prompt: str,
    user_prompt: str,
    *,
    model_provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    retry: int = 3,
    json_output: bool = False,
):
    func = None
    args: dict = {}

    if model_provider == "openai":
        func = run_openai_llm
        args = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_name": model_name,
            "json_output": json_output
        }
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    for attempt in range(retry):
        try:
            return func(**args)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(5)
    raise RuntimeError(
        f"All {retry} attempts failed for model provider: {model_provider}"
    )


def run_llm_structured(
    system_prompt: str,
    user_prompt: str,
    *,
    response_model: BaseModel,
    model_name: str = "gpt-4o-mini",
    retry: int = 3,
):
    # Prepare payloads
    client = openai.OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(retry):
        try:
            llm_response = client.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=response_model
            )
            return llm_response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(5)
    raise RuntimeError(f"All {retry} attempts failed")
