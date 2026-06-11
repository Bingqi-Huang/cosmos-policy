"""Create a CPU-only contact sheet for pair future-frame QA."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any


IMAGE_KEYS = [
    ("current_img_a_path", "A current"),
    ("current_img_b_path", "B current"),
    ("future_img_a_path", "A future"),
    ("future_img_b_path", "B future"),
]


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(repo_root: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else repo_root / path


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int, stride: int) -> list[dict[str, Any]]:
    if stride > 0:
        return rows[::stride][:count]
    rng = random.Random(seed)
    if len(rows) <= count:
        return rows
    indices = sorted(rng.sample(range(len(rows)), count))
    return [rows[i] for i in indices]


def load_tile(path: pathlib.Path, size: int, label: str):
    from PIL import Image, ImageDraw

    if path.exists():
        img = Image.open(path).convert("RGB").resize((size, size))
    else:
        img = Image.new("RGB", (size, size), (80, 80, 80))
    draw = ImageDraw.Draw(img)
    text_bg = (0, 0, 0)
    draw.rectangle((0, 0, size, 18), fill=text_bg)
    draw.text((4, 3), label[:48], fill=(255, 255, 255))
    if not path.exists():
        draw.text((4, size // 2 - 8), "MISSING", fill=(255, 80, 80))
    return img


def make_contact_sheet(rows: list[dict[str, Any]], repo_root: pathlib.Path, output: pathlib.Path, tile_size: int) -> None:
    from PIL import Image, ImageDraw

    label_h = 44
    pad = 8
    cols = len(IMAGE_KEYS)
    rows_n = len(rows)
    width = cols * tile_size + (cols + 1) * pad
    height = rows_n * (tile_size + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        y = pad + row_idx * (tile_size + label_h + pad)
        pair_id = str(row.get("pair_id", "<missing>"))
        meta = (
            f"{pair_id} | split={row.get('split')} | suite={row.get('suite')} | "
            f"dt={int(row.get('future_timestep', 0)) - int(row.get('timestep', 0))}"
        )
        draw.text((pad, y), meta[:160], fill=(0, 0, 0))
        for col, (key, title) in enumerate(IMAGE_KEYS):
            x = pad + col * (tile_size + pad)
            raw = str(row.get(key, ""))
            path = resolve(repo_root, raw) if raw else pathlib.Path("<missing>")
            tile = load_tile(path, tile_size, f"{title}: {path.name}")
            sheet.paste(tile, (x, y + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"[contact-sheet] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stride", type=int, default=0, help="Use every Nth row instead of random sampling when >0")
    parser.add_argument("--tile-size", type=int, default=160)
    args = parser.parse_args()

    rows = read_jsonl(pathlib.Path(args.manifest))
    chosen = sample_rows(rows, count=args.sample_count, seed=args.seed, stride=args.stride)
    make_contact_sheet(chosen, repo_root=pathlib.Path(args.repo_root).resolve(), output=pathlib.Path(args.output), tile_size=args.tile_size)


if __name__ == "__main__":
    main()
