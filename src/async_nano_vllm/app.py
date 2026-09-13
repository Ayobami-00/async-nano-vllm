import time
import uuid
import threading
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

from nanovllm import LLM, SamplingParams

MODEL_PATH = "/models/Qwen3-0.6B"
MODEL_NAME = "Qwen/Qwen3-0.6B"


app = FastAPI()

llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
)

engine_lock = Lock()


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.1


@app.post("/v1/completions")
def completion(request: CompletionRequest):
    request_id = f"cmpl-{uuid.uuid4().hex}"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    request_start = time.perf_counter()

    with engine_lock:

        engine_start = time.perf_counter()

        output = llm.generate(
            [request.prompt],
            sampling_params,
            use_tqdm=False,
        )[0]

        engine_end = time.perf_counter()

    request_end = time.perf_counter()

    return {
        "id": request_id,
        "object": "text_completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": output["text"],
                "finish_reason": "stop",
            }
        ],
        "_debug": {
            "engine_wait_ms": (engine_start - request_start) * 1000,
            "generation_time_ms": (engine_end - engine_start) * 1000,
            "handler_time_ms": (request_end - request_start) * 1000,
            "output_tokens": len(output["token_ids"]),
        },
    }


@app.get("/probe")
def probe():

    thread_id = threading.get_ident()

    start = time.perf_counter()

    print(
        f"START thread={thread_id} " f"time={start:.6f}",
    )

    time.sleep(2)

    end = time.perf_counter()

    print(
        f"END   thread={thread_id} " f"time={end:.6f}",
    )

    return {
        "thread_id": thread_id,
        "duration_ms": (end - start) * 1000,
    }
