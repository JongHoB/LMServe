import argparse
import time

import aiohttp
import asyncio
import tqdm
import numpy as np
import requests

from typing import List
from collections.abc import Callable

from benchmark_utils import ReqResult, summarize_results, save_results
from generator import (generate_requests, generate_radom_requests,
                       generate_trace, APIRequest, APIResponse,
                       APIResponseWithTime, supported_dataset_names)

background_tasks = set()


async def send_request(url: str, request: APIRequest, pbar):
    timeout = aiohttp.ClientTimeout(total=20 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            async with session.post(f"{url}/generate",
                                    json=request) as response:
                if response.status == 200 or response.status == 201:
                    data: APIResponse = await response.json()
                    end_time = time.time()
                    return APIResponseWithTime(
                        start_time=start_time,
                        end_time=end_time,
                        **data,
                    )
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


def clear_cache(url: str):
    print("Clearing server-side KV Cache.")
    requests.post(f"{url}/clear_cache")
    print("Cache cleared successfully.")


def main(args):
    print(args)

    url = f"http://{args.host}:{args.port}"

    num_requests = args.num_requests + args.num_padding_requests
    num_benchmark_reqs = args.num_requests

    intervals = np.random.exponential(1.0 / args.rate, size=num_requests)

    requests: List[APIRequest] = []
    if args.dataset is not None:
        if args.use_time:
            (requests, intervals) = generate_trace(
                dataset_name=args.dataset,
                tokenizer_name=args.tokenizer,
                num_requests=num_requests,
            )
        else:
            requests = generate_requests(
                dataset_name=args.dataset,
                tokenizer_name=args.tokenizer,
                num_requests=num_requests,
                max_seq_len=args.max_seq_len,
                num_samples=args.num_samples,
                ignore_eos=not args.disable_ignore_eos,
            )
    elif args.input_len is not None and args.output_len is not None:
        requests = generate_radom_requests(
            tokenizer_name=args.tokenizer,
            max_input_len=args.input_len,
            max_output_len=args.output_len,
            num_requests=num_requests,
            max_seq_len=args.max_seq_len,
            num_samples=args.num_samples,
        )
    else:
        raise RuntimeError(
            "Invalid configuration: you must either provide a '--dataset'"
            "or specify both '--input-len' and '--output-len' arguments.")

    # Clear server-side cache before running benchmark.
    clear_cache(url)

    responses: APIResponse = []

    start_time = time.time()
    with tqdm.tqdm(total=len(requests)) as pbar:

        def callback(future: asyncio.Future):
            pbar.update(1)
            res = future.result()
            if pbar.n <= num_benchmark_reqs:
                if res is not None:
                    responses.append(res)

        asyncio.run(
            run_client(
                url=url,
                requests=requests,
                num_benchmark_reqs=num_benchmark_reqs,
                intervals=intervals,
                pbar=pbar,
                cb=callback,
            ))

    end_time = responses[-1]['end_time']
    elapsed_time = end_time - start_time

    print(f"\nBenchmark time: {elapsed_time:.2f} s")

    results = [ReqResult.from_response(r) for r in responses]

    benchmark_result = summarize_results(results, elapsed_time)
    # Print benchmark result
    benchmark_result.print()

    if args.result_path is not None and len(results) > 0:
        save_results(results, benchmark_result, base_dir=args.result_path)

    if args.print_output_text:
        outputs = sorted(responses, key=lambda r: r['output_len'])
        print("Below are the generated text of the processed requests:")
        for i, output in enumerate(outputs):
            print("### Generated output for request {}: {}\n".format(
                i, output['output_text']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--dataset", type=str, choices=supported_dataset_names)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--print-output-text", action="store_true")
    parser.add_argument("--num-padding-requests", type=int, default=32)
    parser.add_argument("--use-time", action="store_true")
    parser.add_argument("--result-path", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
