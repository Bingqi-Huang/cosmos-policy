#!/usr/bin/env python3
"""Backfill and optionally tail Cosmos training metrics from run.log into W&B."""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import wandb


ITER_RE = re.compile(
    r"\]\s+(?P<step>\d+)\s+:\s+iter_speed\s+(?P<iter_speed>[0-9.]+)\s+seconds per iteration\s+\|\s+(?P<metrics>.*)$"
)
ITER_ONE_RE = re.compile(
    r"\]\s+Iteration\s+(?P<step>\d+):.*?\|\s+Loss:\s+(?P<loss>[0-9.eE+-]+)\s+\|\s+(?P<metrics>.*)$"
)
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):\s*(?P<value>[-+0-9.eE]+)")
DEVICE_ROW_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z0-9_./-]+)\s+"
    r"(?P<avg>[-+0-9.eE]+)\s+"
    r"(?P<max>[-+0-9.eE]+)\s+"
    r"(?P<min>[-+0-9.eE]+)\s*$"
)


def parse_metric_tail(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in METRIC_RE.finditer(text):
        metrics[match.group("key")] = float(match.group("value"))
    return metrics


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def maybe_flush(step: int | None, metrics: dict[str, float], flushed: set[int]) -> None:
    if step is None or not metrics or step in flushed:
        return
    wandb.log(metrics, step=step)
    flushed.add(step)


def stream_metrics(log_path: Path, follow: bool, poll_seconds: float) -> None:
    pending_step: int | None = None
    pending_metrics: dict[str, float] = {}
    flushed_steps: set[int] = set()
    in_device_block = False

    with log_path.open("r", errors="replace") as file:
        while True:
            line = file.readline()
            if not line:
                maybe_flush(pending_step, pending_metrics, flushed_steps)
                if not follow:
                    break
                time.sleep(poll_seconds)
                continue

            iter_match = ITER_RE.search(line)
            iter_one_match = ITER_ONE_RE.search(line)
            if iter_match or iter_one_match:
                maybe_flush(pending_step, pending_metrics, flushed_steps)
                match = iter_match or iter_one_match
                assert match is not None
                pending_step = int(match.group("step"))
                pending_metrics = parse_metric_tail(match.group("metrics"))
                if iter_match:
                    pending_metrics["iter_speed_sec"] = float(iter_match.group("iter_speed"))
                if iter_one_match:
                    pending_metrics["Loss"] = float(iter_one_match.group("loss"))
                in_device_block = False
                continue

            if "DeviceMonitor Stats:" in line and pending_step is not None:
                in_device_block = True
                continue

            if in_device_block:
                row = DEVICE_ROW_RE.match(line)
                if row:
                    key = row.group("key")
                    pending_metrics[f"device/{key}_avg"] = float(row.group("avg"))
                    pending_metrics[f"device/{key}_max"] = float(row.group("max"))
                    pending_metrics[f"device/{key}_min"] = float(row.group("min"))
                elif line.strip() == "":
                    in_device_block = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project", default="cosmos_policy")
    parser.add_argument("--group", default="cosmos_v2_finetune")
    parser.add_argument("--name", default="phase1_scene_only_formal_6gpu_b30_from_libero_ckpt")
    parser.add_argument("--id", default=None)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    run_id = args.id or read_text(args.run_dir / "wandb_id.txt")
    wandb.init(
        id=run_id,
        project=args.project,
        group=args.group,
        name=args.name,
        dir=os.fspath(args.run_dir),
        resume="allow",
        mode="online",
        config={"source": "run.log backfill", "log_path": os.fspath(args.log)},
    )
    stream_metrics(args.log, args.follow, args.poll_seconds)
    wandb.finish()


if __name__ == "__main__":
    main()
