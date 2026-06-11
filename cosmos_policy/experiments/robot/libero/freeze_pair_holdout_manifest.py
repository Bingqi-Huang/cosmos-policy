"""Freeze the A2 held-out pair manifest (paper_outline LOCKED DECISION 8).

The pair renderer already splits demos into train/val at the demo level
(``--val-demo-fraction``); the merged ``libero_pair_future_manifest_val.jsonl``
is therefore a held-out pair set that the training manifest never touches.
This script snapshots it into a frozen copy with a SHA256 sidecar so that the
"fixed held-out pair set, frozen at Phase-2 completion" property of the A2
shrinkage measurement is auditable: every later shrinkage evaluation must
verify the hash before measuring.

CPU-only; no GPU or torch import.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-manifest",
        default="outputs/phase2/pair_future_frames/libero_pair_future_manifest_val.jsonl",
        help="Merged val (held-out demo) pair manifest produced by merge_pair_future_shards.sh",
    )
    parser.add_argument(
        "--frozen-out",
        default="outputs/phase2/pair_future_frames/libero_pair_future_manifest_holdout_frozen.jsonl",
        help="Frozen held-out manifest path (must not already exist unless --force)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing frozen manifest (discouraged)")
    args = parser.parse_args()

    src = pathlib.Path(args.val_manifest)
    dst = pathlib.Path(args.frozen_out)
    sidecar = dst.with_suffix(".freeze.json")

    if dst.exists() and not args.force:
        raise SystemExit(
            f"Frozen held-out manifest already exists: {dst}\n"
            "LOCKED DECISION 8 forbids re-freezing; use the existing file (or --force only with researcher approval)."
        )

    rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = {json.loads(line).get("split") for line in rows}
    if splits - {"val"}:
        raise SystemExit(f"Refusing to freeze: val manifest contains non-val splits {splits}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    meta = {
        "source": str(src),
        "frozen": str(dst),
        "sha256": sha256_of(dst),
        "num_rows": len(rows),
        "frozen_at": datetime.datetime.now().astimezone().isoformat(),
        "protocol": "paper_outline LOCKED DECISION 8: fixed held-out pair set, per-checkpoint latent-space "
        "shrinkage ratio, deterministic shared-(sigma, n) evaluation, per-sigma-bin reporting, "
        "5th-percentile denominator floor frozen once.",
    }
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
