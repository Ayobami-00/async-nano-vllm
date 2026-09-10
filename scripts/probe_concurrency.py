from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import requests

URL = "http://localhost:8000/probe"


def send_request():
    start = perf_counter()

    response = requests.get(
        URL,
        timeout=30,
    )

    end = perf_counter()

    response.raise_for_status()

    return {
        **response.json(),
        "client_ms": (end - start) * 1000,
    }


benchmark_start = perf_counter()

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(
        executor.map(
            lambda _: send_request(),
            range(4),
        )
    )

benchmark_end = perf_counter()

for result in results:
    print(result)

print(f"Total wall time: " f"{benchmark_end - benchmark_start:.2f}s")
