from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import requests


URL = "http://localhost:8000/v1/completions"

PAYLOAD = {
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello, Nano-vLLM.",
    "max_tokens": 128,
    "temperature": 0.1,
}


def send_request(index):
    start = perf_counter()

    response = requests.post(
        URL,
        json=PAYLOAD,
        timeout=120,
    )

    end = perf_counter()

    response.raise_for_status()

    body = response.json()

    return {
        "request": index,
        "client_e2e_ms": (end - start) * 1000,
        "engine_wait_ms": body["_debug"]["engine_wait_ms"],
        "generation_ms": body["_debug"]["generation_time_ms"],
    }


benchmark_start = perf_counter()

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(
        executor.map(
            send_request,
            range(4),
        )
    )

benchmark_end = perf_counter()

for result in results:
    print(result)

print(
    f"Total wall time: "
    f"{benchmark_end - benchmark_start:.2f}s"
)