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
    result_future: asyncio.Future


incoming_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
pending_results: dict[int, asyncio.Future] = {}


async def engine_loop():

    while True:

        if llm.is_finished():

            request = await incoming_queue.get()

            print(
                f"admitted request={request.request_id} "
                f"queue_depth={incoming_queue.qsize()}"
            )

            seq_id = llm.add_request(
                request.prompt,
                request.sampling_params,
            )

            pending_results[seq_id] = request.result_future

            incoming_queue.task_done()

        while True:
            try:
                request = incoming_queue.get_nowait()

            except asyncio.QueueEmpty:
                break

            print(
                f"admitted request={request.request_id} "
                f"queue_depth={incoming_queue.qsize()}"
            )

            seq_id = llm.add_request(
                request.prompt,
                request.sampling_params,
            )

            pending_results[seq_id] = request.result_future

            incoming_queue.task_done()

        outputs, _ = llm.step()

        print(
            f"external_queue={incoming_queue.qsize()} "
            f"engine_finished={llm.is_finished()}"
        )

        for seq_id, token_ids in outputs:
            future = pending_results.pop(seq_id)
            text = llm.tokenizer.decode(token_ids)

            print(f"completed seq={seq_id}: {text!r}")

            if not future.cancelled():

                future.set_result(
                    {
                        "text": text,
                        "token_ids": token_ids,
                    }
                )

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
async def completion(request: CompletionRequest):
    request_id = f"cmpl-{uuid.uuid4().hex}"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    loop = asyncio.get_running_loop()
    result_future = loop.create_future()

    inference_request = InferenceRequest(
        request_id=request_id,
        prompt=request.prompt,
        sampling_params=sampling_params,
        arrival_time=time.perf_counter(),
    )

    await incoming_queue.put(inference_request)

    result = await result_future

    return {
        "id": request_id,
        "object": "text_completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": result["text"],
                "finish_reason": "stop",
            }
        ],
    }
