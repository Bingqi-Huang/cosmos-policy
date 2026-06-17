"""Generate a nominal-camera task list from a LIBERO-Plus camera task list.

The dissociation Delta(c) needs an unperturbed (nominal-camera) reference bin. For each base
task in the perturbed list this emits "<base>_view_0_0_100_0_0_initstate_0" — zero horizon/
vertical/end-rotation and scale 100, i.e. the nominal agentview (classified as nominal, not
C1/C2/C3). Run through the same run_camera_task path so the predicted-future + state-matched
GT frames are captured identically to the perturbed bins. Pure stdlib; CPU-testable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SUFFIX = re.compile(r"_view_-?\d+_-?\d+_\d+_-?\d+_-?\d+_initstate_\d+$")


def base_tasks(camera_task_names: list[str]) -> list[str]:
    """Unique base task names (perturbation suffix stripped), order-preserving."""
    seen, bases = set(), []
    for name in camera_task_names:
        base = _SUFFIX.sub("", name)
        if base not in seen:
            seen.add(base)
            bases.append(base)
    return bases


def nominal_task_names(camera_task_names: list[str]) -> list[str]:
    return [f"{base}_view_0_0_100_0_0_initstate_0" for base in base_tasks(camera_task_names)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate nominal-camera task names")
    ap.add_argument("--camera_tasks_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = json.loads(Path(args.camera_tasks_file).read_text(encoding="utf-8"))
    nominal = nominal_task_names(names)
    Path(args.out).write_text(json.dumps(nominal, indent=2), encoding="utf-8")
    print(f"nominal tasks: {len(nominal)} -> {args.out}")


if __name__ == "__main__":
    main()
