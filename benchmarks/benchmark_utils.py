import csv
import numpy as np
import logging

from pathlib import Path
from typing import List
from generator import APIResponse


def save_request_stats(
    outputs: List[APIResponse],
    stem: str = "request_stats",
    base_dir: str | Path = ".",
) -> None:

    if not outputs:
        return

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / f"{stem}.csv"

    if candidate.exists():
        suffix = 1
        while True:
            candidate = base_dir / f"{stem}_{suffix}.csv"
            if not candidate.exists():
                break

            suffix += 1

    csv_path = candidate

    outputs.sort(key=lambda x: x['start_time'])
    base_start_time = outputs[0]['start_time']

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "request#",
            "start_time(s)",
            "input_len",
            "output_len",
            "ttft(ms)",
            "tpot_p50(ms)",
            "tpot_p90(ms)",
            "tpot_p99(ms)",
        ])

        for i, out in enumerate(outputs):
            start_time = out["start_time"] - base_start_time
            input_len = len(out["token_ids"])
            output_len = out["output_len"]
            token_lats = out["token_latencies"]

            if output_len < 2:
                logging.warning(
                    f"Skipping request record {i}: output length < 2")
                continue

            ttft = token_lats[0]
            tpot_p50, tpot_p90, tpot_p99 = np.percentile(
                token_lats[1:],
                q=[50, 90, 99],
                method="closest_observation",
            )

            writer.writerow([
                i,
                round(start_time, 3),
                input_len,
                output_len,
                round(ttft * 1000, 4),
                round(tpot_p50 * 1000, 4),
                round(tpot_p90 * 1000, 4),
                round(tpot_p99 * 1000, 4),
            ])

    print(f"Save request stats to {csv_path}")
