"""Generate a compact Markdown report from SCVC sanity-ladder JSON outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def status_mark(status: str) -> str:
    if status == "pass":
        return "PASS"
    if status == "missing":
        return "MISSING"
    if status in {"pending", "skipped"}:
        return status.upper()
    return "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", required=True)
    parser.add_argument("--gpu-json", default="")
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    contract = load_json(pathlib.Path(args.contract_json))
    gpu = load_json(pathlib.Path(args.gpu_json)) if args.gpu_json else {"status": "pending"}

    lines = [
        "# SCVC Sanity Ladder",
        "",
        "| Check | Status | Notes |",
        "|---|---:|---|",
        f"| CPU pair/batch contract | {status_mark(str(contract.get('status')))} | `{args.contract_json}` |",
        f"| GPU equivalence/memory/overfit ladder | {status_mark(str(gpu.get('status')))} | "
        f"{'not run' if not args.gpu_json else '`' + args.gpu_json + '`'} |",
        "",
        "## CPU Contract Summary",
        "",
        "```json",
        json.dumps(contract, indent=2, sort_keys=True),
        "```",
    ]
    if args.gpu_json:
        lines.extend(["", "## GPU Ladder Summary", "", "```json", json.dumps(gpu, indent=2, sort_keys=True), "```"])
    out = pathlib.Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[sanity] wrote {out}")


if __name__ == "__main__":
    main()
