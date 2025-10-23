import aiohttp
import asyncio
import time
import re

from typing import List, TypedDict, Optional, Callable
from queue import Queue, Empty
from itertools import count
from loguru import logger


atomic_counter = count(0)


class APIRequest(TypedDict):
    session_id: Optional[str]
    prompt: str
    num_samples: int
    max_output_len: Optional[int]
    ignore_eos: bool
    disable_cache: bool = True


class APIResponse(TypedDict):
    token_ids: List[int]
    output_text: str
    output_len: int
    token_latencies: List[float]


class APIResponseWithTime(TypedDict):
    start_time: float = None
    end_time: float = None
    prompt: str
    token_ids: List[int]
    output_text: str
    output_len: int
    token_latencies: List[float]


class APIRequestSequence(TypedDict):
    requests: List[APIRequest]
    output_texts: List[str]


async def send_request(url: str, request: APIRequest):
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
                        prompt=request['prompt'],
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


BEGIN_SIGNAL = "### "
END_SIGNAL = "\n"


async def send_requests(
    id: str,
    queue: Queue[APIRequestSequence],
    url: str,
    num_requests: int,
    num_skip_turns: int,
    event: asyncio.Event,
    cb: Callable,
):
    turn_count = 0
    while True:
        try:
            req_seq = queue.get(block=False)
        except Empty:
            logger.warning(
                f"[Client {id}] No task in queue. Terminating send loop.")
            break

        cum_prompt: str = ""
        prev_output_text = ""
        for i, request in enumerate(req_seq['requests']):
            prompt = (BEGIN_SIGNAL + "user:\n" + request['prompt'] +
                      END_SIGNAL + BEGIN_SIGNAL + "assistant:\n")
            if not cum_prompt:
                new_prompt_segment = prompt
            else:
                new_prompt_segment = (prev_output_text + END_SIGNAL +
                                      prompt)

            cum_prompt += new_prompt_segment
            request['prompt'] = cum_prompt

            turn_count += 1
            if turn_count < num_skip_turns:
                prev_output_text = req_seq['output_texts'][i]
                continue
            elif turn_count == num_skip_turns:
                logger.trace(f"[Client {id}] Starting from turn {i+1}")

            tokens = re.findall(r'\w+', new_prompt_segment)
            try:
                await asyncio.wait_for(event.wait(), timeout=len(tokens) * 0.2)
            except asyncio.TimeoutError:
                pass

            if next(atomic_counter) >= num_requests:
                event.set()
                return

            task = asyncio.create_task(send_request(url, request))
            task.add_done_callback(cb)

            response = await task

            prev_output_text = response['output_text']


async def run_client(
    url: str,
    requests: List[APIRequest],
    intervals: List[float],
    cb: Callable[[asyncio.Future], None],
):
    background_tasks = set()
    for request, interval in zip(requests, intervals):
        task = asyncio.create_task(send_request(url, request))
        background_tasks.add(task)
        task.add_done_callback(cb)

        await asyncio.sleep(interval)

    done, pending = await asyncio.wait(background_tasks)


async def run_multi_turn_client(
    url: str,
    request_sequences: List[APIRequestSequence],
    num_requests: int,
    num_clients: int,
    num_skip_turns: int,
    cb: Callable[[asyncio.Future], None],
):
    task_queue = Queue()
    for req_seq in request_sequences:
        task_queue.put(req_seq)

    global atomic_counter
    atomic_counter = count(0)

    event = asyncio.Event()

    clients = [
        asyncio.create_task(
            send_requests(str(i), task_queue, url, num_requests,
                          num_skip_turns, event, cb))
        for i in range(num_clients)
    ]

    await asyncio.gather(*clients)


def clear_cache(url: str):
    import requests
    print("Clearing server-side KV Cache.")
    requests.post(f"{url}/clear_cache")
    print("Cache cleared successfully.")
