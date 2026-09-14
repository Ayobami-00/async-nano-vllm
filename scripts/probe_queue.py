import asyncio
from time import perf_counter

import httpx


URL = "http://localhost:8000/v1/submit"

PAYLOAD = {
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello, Nano-vLLM.",
    "max_tokens": 128,
    "temperature": 0.1,
}


async def send_request(
    client: httpx.AsyncClient,
    request_id: int,
):
    response = await client.post(
        URL,
        json=PAYLOAD,
    )

    response.raise_for_status()

    return {
        "request": request_id,
        **response.json(),
    }


async def main():
    num_requests = 10

    start = perf_counter()

    async with httpx.AsyncClient(
        timeout=30,
    ) as client:
        results = await asyncio.gather(
            *[
                send_request(client, i)
                for i in range(num_requests)
            ]
        )

    elapsed = perf_counter() - start

    for result in results:
        print(result)

    print(
        f"Admitted requests: {len(results)}"
    )

    print(
        f"Submission rate: "
        f"{len(results) / elapsed:.2f} req/s"
    )

    print(
        f"Final observed queue depth: "
        f"{max(r['queue_depth'] for r in results)}"
    )


asyncio.run(main())