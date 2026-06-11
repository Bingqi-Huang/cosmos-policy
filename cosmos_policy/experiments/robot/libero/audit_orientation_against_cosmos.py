"""Compare a rendered frame against a Cosmos/reference frame under simple orientation transforms."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Callable


def transform_fns():
    from PIL import Image

    return {
        "none": lambda img: img,
        "flipud": lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        "fliplr": lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "rot180": lambda img: img.transpose(Image.Transpose.ROTATE_180),
        "transpose": lambda img: img.transpose(Image.Transpose.TRANSPOSE),
        "transverse": lambda img: img.transpose(Image.Transpose.TRANSVERSE),
    }


def load_rgb(path: pathlib.Path, size: int):
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size))


def mae(a, b) -> float:
    import numpy as np

    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def make_contact_sheet(reference, rendered, candidates: dict[str, object], output: pathlib.Path, size: int) -> None:
    from PIL import Image, ImageDraw

    pad = 8
    label_h = 24
    cols = 2 + len(candidates)
    width = cols * size + (cols + 1) * pad
    height = size + label_h + 2 * pad
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    tiles = [("reference", reference), ("rendered", rendered)] + list(candidates.items())
    for idx, (label, img) in enumerate(tiles):
        x = pad + idx * (size + pad)
        draw.text((x, pad), str(label)[:30], fill=(0, 0, 0))
        sheet.paste(img, (x, pad + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Cosmos/source image path")
    parser.add_argument("--rendered", required=True, help="Renderer output image path")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-contact-sheet", default="")
    args = parser.parse_args()

    reference_path = pathlib.Path(args.reference)
    rendered_path = pathlib.Path(args.rendered)
    reference = load_rgb(reference_path, args.size)
    rendered = load_rgb(rendered_path, args.size)

    scores = {}
    transformed = {}
    for name, fn in transform_fns().items():
        candidate = fn(rendered.copy())
        transformed[name] = candidate
        scores[name] = mae(reference, candidate)
    best = min(scores, key=scores.get)
    report = {
        "reference": str(reference_path),
        "rendered": str(rendered_path),
        "size": args.size,
        "best_transform": best,
        "scores_mae": dict(sorted(scores.items(), key=lambda item: item[1])),
    }
    out = pathlib.Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_contact_sheet:
        make_contact_sheet(reference, rendered, transformed, pathlib.Path(args.output_contact_sheet), args.size)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
