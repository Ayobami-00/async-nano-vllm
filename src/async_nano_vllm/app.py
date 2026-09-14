import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from pydantic import BaseModel

from nanovllm import LLM, SamplingParams

MODEL_PATH = "/models/Qwen3-0.6B"
MODEL_NAME = "Qwen/Qwen3-0.6B"


llm = LLM(
    MODEL_PATH,
    enforce_eager=True,
    tensor_parallel_size=1,
)


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.1


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float


incoming_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()


async def engine_loop():

    while True:

        if llm.is_finished():

            request = await incoming_queue.get()

            llm.add_request(
                request.prompt,
                request.sampling_params,
            )

            incoming_queue.task_done()

        while True:
            try:
                request = incoming_queue.get_nowait()

            except asyncio.QueueEmpty:
                break

            llm.add_request(
                request.prompt,
                request.sampling_params,
            )

            incoming_queue.task_done()

        outputs, _ = llm.step()

        for seq_id, token_ids in outputs:
            text = llm.tokenizer.decode(token_ids)
            print(f"completed seq={seq_id}: {text!r}")

        await asyncio.sleep(0)


@asynccontextmanager
async def lifespan(app: FastAPI):

    engine_task = asyncio.create_task(
        engine_loop(),
        name="nano-vllm-engine-loop",
    )

    try:
        yield
    finally:
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


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


@app.post("/v1/submit", status_code=202)
async def submit(request: CompletionRequest):

    request_id = f"cmpl-{uuid.uuid4().hex}"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    inference_request = InferenceRequest(
        request_id=request_id,
        prompt=request.prompt,
        sampling_params=sampling_params,
        arrival_time=time.perf_counter(),
    )

    await incoming_queue.put(inference_request)

    return {
        "id": request_id,
        "status": "queued",
        "queue_depth": incoming_queue.qsize(),
    }
