import csv
import numpy as np
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from generator import APIResponseWithTime


@dataclass
class ReqResult:
    start_time: int
    total_len: int
    output_len: int
    latency: float
    ttft: float
    tpot: Optional[float]
    second_tok_lat: Optional[float]

    @classmethod
    def from_response(cls, res: APIResponseWithTime) -> "ReqResult":
        tok_lats = res['token_latencies']
        output_len = len(tok_lats)

        return cls(
            start_time=res['start_time'],
            total_len=len(res['token_ids']),
            output_len=output_len,
            latency=sum(tok_lats),
            ttft=tok_lats[0],
            tpot=sum(tok_lats[1:]) / output_len if output_len > 1 else None,
            second_tok_lat=tok_lats[1] if output_len > 1 else None,
        )


@dataclass
class BenchmarkResult:
    req_tput: float
    tok_tput: float
    out_tok_tput: float
    avg_req_lat: float

    # The lists below include P50, P90, and P99 tail latencies.
    ttft_tails: List[float]
    tpot_tails: List[float]
    second_tok_lat_tails: List[float]

    # SLO attainment
    req_gput: float
    ttft_gput: float
    tpot_gput: float

    req_gput_pct: float
    ttft_gput_pct: float
    tpot_gput_pct: float

    def print(self):
        print(f"Throughput (request): {self.req_tput:.2f} reqs/s")
        print(f"Throughput (token): {self.tok_tput:.2f} tokens/s")
        print(f"Throughput (output token): {self.out_tok_tput:.2f} tokens/s")
        print(f"Avg request latency: {self.avg_req_lat:.2f} s")

        ttft_tails_ms = [t * 1000 for t in self.ttft_tails]
        print("TTFT:")
        print("P50: {:.2f} ms, P90: {:.2f} ms, P99: {:.2f} ms".format(
            *ttft_tails_ms))

        tpot_tails_ms = [t * 1000 for t in self.tpot_tails]
        print("TPOT:")
        print("P50: {:.2f} ms, P90: {:.2f} ms, P99: {:.2f} ms".format(
            *tpot_tails_ms))

        second_tok_lat_tails_ms = [t * 1000 for t in self.second_tok_lat_tails]
        print("Second token latency:")
        print("P50: {:.2f} ms, P90: {:.2f} ms, P99: {:.2f} ms".format(
            *second_tok_lat_tails_ms))

        print(f"Goodput (request): {self.req_gput:.2f} req/s "
              f"({self.req_gput_pct:.2f} %)")
        print(f"Goodput (TTFT): {self.ttft_gput:.2f} req/s "
              f"({self.ttft_gput_pct:.2f} %)")
        print(f"Goodput (TPOT): {self.tpot_gput:.2f} req/s "
              f"({self.tpot_gput_pct:.2f} %)")

        print("")

    def save(self, dir: str | Path):
        file_path = dir / "benchmark_result.csv"

        metrics = {
            "request_throughput": self.req_tput,
            "token_throughput": self.tok_tput,
            "output_token_throughput": self.out_tok_tput,
            "avg_request_latency": self.avg_req_lat,
            "ttft_p50_ms": self.ttft_tails[0] * 1000,
            "ttft_p90_ms": self.ttft_tails[1] * 1000,
            "ttft_p99_ms": self.ttft_tails[2] * 1000,
            "tpot_p50_ms": self.tpot_tails[0] * 1000,
            "tpot_p90_ms": self.tpot_tails[1] * 1000,
            "tpot_p99_ms": self.tpot_tails[2] * 1000,
            "second_token_latency_p50_ms": self.second_tok_lat_tails[0] * 1000,
            "second_token_latency_p90_ms": self.second_tok_lat_tails[1] * 1000,
            "second_token_latency_p99_ms": self.second_tok_lat_tails[2] * 1000,
            "request_goodput": self.req_gput,
            "ttft_goodput": self.ttft_gput,
            "tpot_goodput": self.tpot_gput,
            "request_goodput_%": self.req_gput_pct,
            "ttft_goodput_%": self.ttft_gput_pct,
            "tpot_goodput_%": self.tpot_gput_pct,
        }

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for name, value in metrics.items():
                writer.writerow([name, value])


def safe_percentiles(values: List[float]) -> List[Optional[float]]:
    try:
        return np.percentile(
            values,
            q=[50, 90, 99],
            method="closest_observation",
        )
    except IndexError:
        return [None, None, None]


def save_request_stats(
    results: List[ReqResult],
    stem: str = "request_stats",
    base_dir: str | Path = ".",
) -> None:

    if not results:
        return

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / f"{stem}.csv"

    if candidate.exists():
        suffix = 1
        while True:
            candidate = base_dir / f"{stem}{suffix}.csv"
            if not candidate.exists():
                break

            suffix += 1

    csv_path = candidate

    results = sorted(results, key=lambda x: x.start_time)
    base_start_time = results[0].start_time

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "request#",
            "start_time_s",
            "input_len",
            "output_len",
            "ttft_ms",
            "tpot_ms",
        ])

        for i, result in enumerate(results):
            start_time = result.start_time - base_start_time
            output_len = result.output_len
            input_len = result.total_len - output_len

            if output_len < 2:
                logging.warning(
                    f"Skipping request record {i}: output length < 2")
                continue

            writer.writerow([
                i,
                round(start_time, 3),
                input_len,
                output_len,
                round(result.ttft * 1000, 2),
                round(result.tpot * 1000, 2),
            ])


def save_request_trace(
    results: List[ReqResult],
    ttft_slo: float = 2,
    tpot_slo: float = 0.2,
    stem: str = "request_trace",
    base_dir: str | Path = ".",
) -> None:

    if not results:
        return

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / f"{stem}.csv"

    if candidate.exists():
        suffix = 1
        while True:
            candidate = base_dir / f"{stem}{suffix}.csv"
            if not candidate.exists():
                break

            suffix += 1

    csv_path = candidate

    results = sorted(results, key=lambda x: x.start_time)
    base_start_time = results[0].start_time
    total_time = (results[-1].start_time - base_start_time)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "num_requests",
            "total_input_len",
            "total_output_len",
            "ttft_p50_ms",
            "ttft_p90_ms",
            "ttft_p99_ms",
            "tpot_p50_ms",
            "tpot_p90_ms",
            "tpot_p99_ms",
            "num_good_requests",
            "num_good_ttft",
            "num_good_tpot",
        ])

        for time in range(int(total_time) + 1):
            num_reqs = 0
            total_input_len = 0
            total_output_len = 0
            ttfts = []
            tpots = []

            num_good_reqs = 0
            num_good_ttft = 0
            num_good_tpot = 0

            # Aggregate request stats for the current time step
            while len(results) > 0:
                start_time = results[0].start_time - base_start_time
                if start_time >= time + 1:
                    break

                res = results.pop(0)

                num_reqs += 1
                output_len = res.output_len
                input_len = res.total_len - output_len

                total_input_len += input_len
                total_output_len += output_len

                ttft = res.ttft
                tpot = res.tpot

                ttft_met = ttft <= ttft_slo
                tpot_met = (tpot is None or tpot <= tpot_slo)

                ttfts.append(ttft)
                if tpot is not None:
                    tpots.append(tpot)

                num_good_reqs += (ttft_met and tpot_met)
                num_good_ttft += ttft_met
                num_good_tpot += tpot_met

            ttft_p50, ttft_p90, ttft_p99 = safe_percentiles(ttfts)
            tpot_p50, tpot_p90, tpot_p99 = safe_percentiles(tpots)

            writer.writerow([
                time,
                num_reqs,
                total_input_len,
                total_output_len,
                round(ttft_p50 * 1000, 2) if ttft_p50 else np.nan,
                round(ttft_p90 * 1000, 2) if ttft_p90 else np.nan,
                round(ttft_p99 * 1000, 2) if ttft_p99 else np.nan,
                round(tpot_p50 * 1000, 2) if tpot_p50 else np.nan,
                round(tpot_p90 * 1000, 2) if tpot_p90 else np.nan,
                round(tpot_p99 * 1000, 2) if tpot_p99 else np.nan,
                num_good_reqs,
                num_good_ttft,
                num_good_tpot,
            ])


def summarize_results(
    results: List[ReqResult],
    elapsed_time: float,
    ttft_slo: float = 2,
    tpot_slo: float = 0.2,
) -> BenchmarkResult:

    num_reqs = len(results)
    req_tput = num_reqs / elapsed_time

    total_num_tokens = sum(r.total_len for r in results)
    tok_tput = total_num_tokens / elapsed_time

    total_num_output_tokens = sum(r.output_len for r in results)
    out_tok_tput = total_num_output_tokens / elapsed_time

    avg_req_lat = np.mean([r.latency for r in results])

    ttfts = [r.ttft for r in results]
    ttft_tails = safe_percentiles(ttfts)

    tpots = [r.tpot for r in results if r.tpot is not None]
    tpot_tails = safe_percentiles(tpots)

    second_tok_lats = [
        r.second_tok_lat for r in results if r.second_tok_lat is not None
    ]
    second_tok_lat_tails = safe_percentiles(second_tok_lats)

    # If the output length is 1, the TPOT value will be None.
    # In that case, count the request as meeting the SLO.
    num_good_reqs = sum(
        r.ttft <= ttft_slo and (r.tpot is None or r.tpot <= tpot_slo)
        for r in results)
    num_good_ttft = sum(r.ttft <= ttft_slo for r in results)
    num_good_tpot = sum(
        (r.tpot is None or r.tpot <= tpot_slo) for r in results)

    req_gput = num_good_reqs / elapsed_time
    ttft_gput = num_good_ttft / elapsed_time
    tpot_gput = num_good_tpot / elapsed_time

    req_gput_pct = (num_good_reqs / num_reqs) * 100
    ttft_gput_pct = (num_good_ttft / num_reqs) * 100
    tpot_gput_pct = (num_good_tpot / num_reqs) * 100

    return BenchmarkResult(
        req_tput=req_tput,
        tok_tput=tok_tput,
        out_tok_tput=out_tok_tput,
        avg_req_lat=avg_req_lat,
        ttft_tails=ttft_tails,
        tpot_tails=tpot_tails,
        second_tok_lat_tails=second_tok_lat_tails,
        req_gput=req_gput,
        ttft_gput=ttft_gput,
        tpot_gput=tpot_gput,
        req_gput_pct=req_gput_pct,
        ttft_gput_pct=ttft_gput_pct,
        tpot_gput_pct=tpot_gput_pct,
    )


def save_results(
    results: List[ReqResult],
    benchmark_result: Optional[BenchmarkResult] = None,
    base_dir: str | Path = "./results",
) -> None:
    if not results:
        return

    candidate = Path(base_dir)
    if candidate.exists():
        suffix = 1
        while True:
            candidate = Path(f"{base_dir}{suffix}")
            if not candidate.exists():
                break

            suffix += 1

    base_dir = candidate
    base_dir.mkdir(parents=True, exist_ok=True)

    benchmark_result.save(base_dir)
    save_request_stats(results, base_dir=base_dir)
    save_request_trace(results, base_dir=base_dir)

    print(f"Save results into {base_dir}")
