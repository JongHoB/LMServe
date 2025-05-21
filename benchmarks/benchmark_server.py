import argparse
import time

import aiohttp
import asyncio
import tqdm
import numpy as np

from typing import List
from collections.abc import Callable

from request import generate_requests, APIRequest, APIResponse

background_tasks = set()


async def send_request(url: str, request: APIRequest, pbar):
    timeout = aiohttp.ClientTimeout(total=20 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            async with session.post(f"{url}/generate",
                                    json=request) as response:
                if response.status == 200 or response.status == 201:
                    data = await response.json()
                    end_time = time.time()
                    data['end_time'] = end_time
                    data['latency'] = (end_time - start_time)
                    return APIResponse(**data)
                else:
                    print(f"Failed: {response.status}")
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return None
        except asyncio.TimeoutError:
            end_time = time.time()
            print("Timeout error: Server did not respons in time "
                  f"over {end_time - start_time:.2f}s")
            return None
        except Exception as e:
            print(f" Unexpected error: {e}")
            return None


async def run_client(
    url: str,
    requests: List[APIRequest],
    num_benchmark_reqs: int,
    intervals: List[float],
    pbar: tqdm.std.tqdm,
    cb: Callable[[asyncio.Future], None],
):
    for request, interval in zip(requests, intervals):
        task = asyncio.create_task(send_request(url, request, pbar))
        background_tasks.add(task)
        task.add_done_callback(cb)

        await asyncio.sleep(interval)

    done, pending = await asyncio.wait(background_tasks)


def main(args):
    print(args)

    # 32 means number of padding requests
    num_requests = args.num_requests + 32

    requests: List[APIRequest] = generate_requests(
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        num_requests=num_requests,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_samples,
    )

    num_benchmark_reqs = args.num_requests
    intervals = np.random.exponential(1.0 / args.rate, size=num_requests)

    url = f"http://{args.host}:{args.port}"

    outputs: APIResponse = []

    start_time = time.time()
    with tqdm.tqdm(total=len(requests)) as pbar:

        def callback(future: asyncio.Future):
            pbar.update(1)
            output = future.result()
            if pbar.n <= num_benchmark_reqs:
                if output is not None:
                    outputs.append(output)

        asyncio.run(
            run_client(
                url=url,
                requests=requests,
                num_benchmark_reqs=num_benchmark_reqs,
                intervals=intervals,
                pbar=pbar,
                cb=callback,
            ))

    end_time = outputs[-1]['end_time']
    elapsed_time = end_time - start_time

    print(f"\nBenchmark time: {elapsed_time:.2f} s")

    print(f"Throughput (request): {len(requests) / elapsed_time:.2f} reqs/s")

    total_num_tokens = sum(len(o['token_ids']) for o in outputs)
    print(
        f"Throughput (token): {total_num_tokens / elapsed_time:.2f} tokens/s")

    total_num_output_tokens = sum(o['output_len'] for o in outputs)
    print("Throughput (output token): "
          f"{total_num_output_tokens / elapsed_time:.2f} tokens/s")

    avg_req_latency = np.mean([o['latency'] for o in outputs])
    print(f"Avg request latency: {avg_req_latency:.2f} s")

    avg_norm_token_latency = np.mean(
        [o['latency'] / o['output_len'] for o in outputs])
    print("Avg normalized output token latency: "
          f"{avg_norm_token_latency:.4f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--tokenizer",
                        type=str,
                        default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
