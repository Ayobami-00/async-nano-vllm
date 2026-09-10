import uuid

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

    output = llm.generate(
        [request.prompt],
        sampling_params,
        use_tqdm=False,
    )[0]

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
    }
