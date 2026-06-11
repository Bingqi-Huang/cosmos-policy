"""Summarize and QA a Phase-2 pair future-frame manifest on CPU."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from statistics import mean
from typing import Any


IMAGE_KEYS = ("current_img_a_path", "current_img_b_path", "future_img_a_path", "future_img_b_path")


def read_jsonl(path: pathlib.Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def resolve(repo_root: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else repo_root / path


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize(rows: list[dict[str, Any]], repo_root: pathlib.Path, check_images: bool) -> dict[str, Any]:
    dt_values = [int(row["future_timestep"]) - int(row["timestep"]) for row in rows if "future_timestep" in row and "timestep" in row]
    pair_ids = Counter(str(row.get("pair_id")) for row in rows)
    missing_images = []
    if check_images:
        for row in rows:
            for key in IMAGE_KEYS:
                if key not in row:
                    missing_images.append(f"<missing-key>:{key}:{row.get('pair_id')}")
                    continue
                path = resolve(repo_root, str(row[key]))
                if not path.exists():
                    missing_images.append(str(path))
    duplicate_pair_ids = [pair_id for pair_id, count in pair_ids.items() if count > 1]
    bad_future = [
        str(row.get("pair_id"))
        for row in rows
        if "future_timestep" in row
        and "timestep" in row
        and "chunk_size" in row
        and (int(row["future_timestep"]) < int(row["timestep"]) or int(row["future_timestep"]) - int(row["timestep"]) > int(row["chunk_size"]))
    ]
    return {
        "num_rows": len(rows),
        "by_split": counter_dict(Counter(row.get("split", "<missing>") for row in rows)),
        "by_suite": counter_dict(Counter(row.get("suite", "<missing>") for row in rows)),
        "by_task": counter_dict(Counter(row.get("task", "<missing>") for row in rows)),
        "by_camera_category": counter_dict(Counter(row.get("camera_category", "<missing>") for row in rows)),
        "by_pair_type": counter_dict(Counter(row.get("pair_type", "<missing>") for row in rows)),
        "future_delta": {
            "min": min(dt_values) if dt_values else None,
            "max": max(dt_values) if dt_values else None,
            "mean": mean(dt_values) if dt_values else None,
            "hist": counter_dict(Counter(dt_values)),
        },
        "num_duplicate_pair_ids": len(duplicate_pair_ids),
        "sample_duplicate_pair_ids": duplicate_pair_ids[:20],
        "num_bad_future_indices": len(bad_future),
        "sample_bad_future_indices": bad_future[:20],
        "num_missing_images": len(missing_images),
        "sample_missing_images": missing_images[:20],
    }


def write_markdown(path: pathlib.Path, report: dict[str, Any]) -> None:
    lines = [
        "# Pair Manifest Summary",
        "",
        f"- Manifest: `{report['manifest']}`",
        f"- Rows checked: {report['summary']['num_rows']}",
        f"- Missing images: {report['summary']['num_missing_images']}",
        f"- Duplicate pair ids: {report['summary']['num_duplicate_pair_ids']}",
        f"- Bad future indices: {report['summary']['num_bad_future_indices']}",
        "",
        "## Counts",
        "",
    ]
    for key in ("by_split", "by_suite", "by_camera_category", "by_pair_type"):
        lines.append(f"### {key}")
        for name, count in report["summary"][key].items():
            lines.append(f"- `{name}`: {count}")
        lines.append("")
    lines.extend(["## Future Delta", "", "```json", json.dumps(report["summary"]["future_delta"], indent=2, sort_keys=True), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    manifest = pathlib.Path(args.manifest)
    repo_root = pathlib.Path(args.repo_root).resolve()
    rows = read_jsonl(manifest, max_rows=args.max_rows)
    report = {
        "manifest": str(manifest),
        "repo_root": str(repo_root),
        "max_rows": args.max_rows,
        "check_images": args.check_images,
        "summary": summarize(rows, repo_root=repo_root, check_images=args.check_images),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(pathlib.Path(args.output_md), report)


if __name__ == "__main__":
    main()
