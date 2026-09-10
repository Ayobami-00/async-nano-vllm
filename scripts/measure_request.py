import time

import requests


URL = "http://localhost:8000/v1/completions"

payload = {
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello, Nano-vLLM.",
    "max_tokens": 128,
    "temperature": 0.1,
}


start = time.perf_counter()

response = requests.post(
    URL,
    json=payload,
    timeout=120,
)

end = time.perf_counter()

response.raise_for_status()

body = response.json()

e2e_ms = (end - start) * 1000
generation_ms = body["_debug"]["generation_time_ms"]
output_tokens = body["_debug"]["output_tokens"]

print(f"E2E latency:       {e2e_ms:.2f} ms")
print(f"Generation time:   {generation_ms:.2f} ms")
print(f"Output tokens:     {output_tokens}")