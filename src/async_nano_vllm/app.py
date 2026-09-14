import asyncio
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json

from collections.abc import AsyncIterator
from fastapi.responses import StreamingResponse
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
class StepOutput:
    seq_id: int
    token_id: int
    finished: bool


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
    output_queue: asyncio.Queue[StepOutput]


incoming_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()

output_queues: dict[
    int,
    asyncio.Queue[StepOutput],
] = {}


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

            output_queues[seq_id] = request.output_queue

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

            output_queues[seq_id] = request.output_queue

            incoming_queue.task_done()

        outputs, _ = llm.step()

        print(
            f"external_queue={incoming_queue.qsize()} "
            f"engine_finished={llm.is_finished()}"
        )

        for output in outputs:
            output_queue = output_queues[output.seq_id]

            await output_queue.put(output)

            if output.finished:
                output_queues.pop(output.seq_id)

        await asyncio.sleep(0)


async def generate(
    request_id: str,
    prompt: str,
    sampling_params: SamplingParams,
) -> AsyncIterator[StepOutput]:

    output_queue: asyncio.Queue[StepOutput] = asyncio.Queue()

    inference_request = InferenceRequest(
        request_id=request_id,
        prompt=prompt,
        sampling_params=sampling_params,
        arrival_time=time.perf_counter(),
        output_queue=output_queue,
    )

    await incoming_queue.put(inference_request)

    while True:

        output = await output_queue.get()

        yield output

        if output.finished:
            break


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

    token_ids = []

    async for output in generate(
        request_id,
        request.prompt,
        sampling_params,
    ):
        token_ids.append(output.token_id)

    text = llm.tokenizer.decode(token_ids)

    return {
        "id": request_id,
        "object": "text_completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
    }
