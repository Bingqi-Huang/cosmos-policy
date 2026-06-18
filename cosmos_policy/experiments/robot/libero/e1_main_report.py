"""E1-main dissociation criterion + report (paper_outline LOCKED DECISION 13).

Combines the two primary readouts into the pre-registered dissociation judgment:

  * action side  : rollout success rate under perturbed cameras, stratified into cells
                   (condition C1/C2/C3 x difficulty level 1-5), reusing the existing
                   camera-eval classification so the cells match generate_camera_report.
  * video side   : camera-conditioned excess-FID per cell + the relative degradation
                   Delta(c) vs the nominal bin (from e1_main_fid). NOTE: the metric is
                   excess-FID, not FVD — Cosmos emits a single predicted future frame per
                   query, so LD13's "excess-FVD" is realised at the frame level. Internal
                   field/arg names keep the *fvd spelling for backward compatibility.

Pre-registered criterion (LD13): over the aggregated severity band where action success
drops by >= 20 pp relative to nominal (cells with >= 100 episodes only), the relative
excess-FID degradation must be <= 25%. LD13's text supports three readings of "the
aggregated severity band" (global all-cell veto, global episode-weighted aggregate, and
per-axis); this module reports ALL three and auto-selects NONE as primary (researcher-only).
Dissociation HOLDS when, in the chosen band, fidelity AND LD13c semantic correctness survive
while action success collapses; without LD13c, verdicts are fidelity-side only / provisional.

The action aggregation reuses the tested helpers in generate_camera_report; the
criterion logic here is pure and CPU-unit-testable. The framing-matrix cell (a/b/c/d)
is only *recorded* — the framing decision is researcher-only (standing rule 10).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Reuse the existing, already-tested camera classification (pure stdlib; no GPU import).
from cosmos_policy.experiments.robot.libero.generate_camera_report import (
    _classify_condition,
    load_task_classification,
    parse_jsonl_files,
)


@dataclass(frozen=True)
class CellKey:
    condition: str  # "C1" | "C2" | "C3"
    level: int  # difficulty 1-5

    def __str__(self) -> str:
        return f"{self.condition}_L{self.level}"


@dataclass
class CellAction:
    key: CellKey
    n_success: int
    n_total: int

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total > 0 else 0.0


@dataclass
class DissociationConfig:
    nominal_success_rate: float  # unperturbed action success (e.g. ID eval SR), in [0, 1]
    action_drop_threshold_pp: float = 20.0  # LD13: >= 20 pp drop defines the severity band
    rel_excess_fvd_threshold: float = 0.25  # LD13: relative excess-FVD degradation <= 25%
    min_episodes_per_cell: int = 100  # LD13: aggregated cells of >= 100 episodes only
    eps: float = 1e-6


@dataclass
class CellVerdict:
    key: str
    n_total: int
    action_success_rate: float
    action_drop_pp: float
    in_severity_band: bool  # action drop >= threshold AND enough episodes
    excess_fvd: Optional[float]
    rel_excess_fvd_degradation: Optional[float]  # Delta(c)
    fidelity_preserved: Optional[bool]  # Delta <= threshold
    semantic_preserved: Optional[bool]  # None until the semantic readout lands


@dataclass
class DissociationResult:
    config: dict
    nominal_excess_fvd: Optional[float]
    cells: list[dict] = field(default_factory=list)
    severity_band_cells: list[str] = field(default_factory=list)
    underpowered_cells: list[str] = field(default_factory=list)
    dissociation_holds: Optional[bool] = None  # == verdicts.global_strict_all_cells (backward-compat alias)
    verdicts: dict = field(default_factory=dict)  # global_strict / global_weighted / per_axis (NONE auto-primary)
    semantic_status: str = "missing"  # missing | provisional ; "verified" is NEVER set without LD13c
    framing_matrix_readout: str = "pending"  # recorded only; researcher decides the framing
    notes: list[str] = field(default_factory=list)


# LIBERO-Plus camera condition labels. Geometry VERIFIED 2026-06-18 against LIBERO-Plus
# libero_tabletop_manipulation._setup_camera (+ scale_distance_from_pivot / rotate_around_y/z):
#   C1 = scale_factor                  -> camera DOLLY: distance scaling along view ray (orientation fixed)
#   C2 = horizon_view + vertical_view  -> ORBITAL viewpoint change: azimuth + elevation (pos AND orientation move)
#   C3 = end_point_rot + end_point_vert-> IN-PLACE reorientation: orientation rotated about Z,Y (position fixed)
# (NOT "pan/tilt": pan/tilt implies a fixed camera position, but C2 orbits the camera to a new vantage.)
AXIS_LABELS = {
    "C1": "dolly (camera distance scaling)",
    "C2": "orbital azimuth/elevation",
    "C3": "in-place reorientation (endpoint rot/vert)",
}


def _axis_of(cell_key: str) -> str:
    return cell_key.split("_")[0]


def _band_fidelity_cells(cells: list[dict]) -> list[dict]:
    """Severity-band cells that have a computable fidelity readout (valid Delta)."""
    return [
        c
        for c in cells
        if c.get("in_severity_band") and c.get("rel_excess_fvd_degradation") is not None
    ]


def _cell_strict_pass(c: dict) -> bool:
    """LD13 one-cell pass: fidelity preserved AND semantic not actively broken (None tolerated)."""
    return (c.get("fidelity_preserved") is True) and (c.get("semantic_preserved") is not False)


def _strict_verdict(band_cells: list[dict]) -> Optional[bool]:
    """All-cells veto over a band cell set; None if no qualifying cells."""
    if not band_cells:
        return None
    return all(_cell_strict_pass(c) for c in band_cells)


def _weighted_verdict(
    band_cells: list[dict], threshold: float
) -> tuple[Optional[float], Optional[bool]]:
    """Episode-weighted mean Delta over a band cell set; verdict = weighted_delta <= threshold."""
    if not band_cells:
        return None, None
    den = sum(c["n_total"] for c in band_cells)
    if den <= 0:
        return None, None
    num = sum(c["n_total"] * c["rel_excess_fvd_degradation"] for c in band_cells)
    wdelta = num / den
    return wdelta, (wdelta <= threshold)


def _compute_multi_verdicts(result: DissociationResult, threshold: float) -> None:
    """Populate result.verdicts with three pre-registration readings of LD13.

    NONE is auto-selected as primary: LD13's text supports the global all-cell veto, the
    global episode-weighted band aggregate, AND a per-axis reading. Surfacing all three is
    deliberate; choosing the primary criterion is a researcher-only decision (standing rule 4).
    All three are FIDELITY-SIDE ONLY; result.semantic_status flags LD13c coverage.
    """
    band = _band_fidelity_cells(result.cells)
    g_strict = _strict_verdict(band)
    g_wdelta, g_weighted = _weighted_verdict(band, threshold)

    per_axis: dict[str, dict] = {}
    for axis in ("C1", "C2", "C3"):
        ab = [c for c in band if _axis_of(c["key"]) == axis]
        s = _strict_verdict(ab)
        wd, wv = _weighted_verdict(ab, threshold)
        per_axis[axis] = {
            "label": AXIS_LABELS[axis],
            "n_band_cells": len(ab),
            "band_cells": [c["key"] for c in ab],
            "fail_cells_strict": [c["key"] for c in ab if not _cell_strict_pass(c)],
            "strict_verdict": s,
            "weighted_delta": wd,
            "weighted_verdict": wv,
            "total_episodes": sum(c["n_total"] for c in ab),
            "semantic_status": result.semantic_status,
        }

    result.verdicts = {
        "_doc": (
            "Three pre-registration readings of LD13 surfaced for the researcher; NONE is "
            "auto-selected as primary. All are FIDELITY-SIDE ONLY (excess-FID); full LD13 "
            "(fidelity AND LD13c semantic correctness) is NOT verified while semantic_status "
            "!= verified, which it is not. dissociation_holds == global_strict_all_cells.verdict."
        ),
        "rel_excess_fid_threshold": threshold,
        "semantic_status": result.semantic_status,
        "global_strict_all_cells": {
            "verdict": g_strict,
            "n_band_cells": len(band),
            "fail_cells": [c["key"] for c in band if not _cell_strict_pass(c)],
        },
        "global_weighted_severity_band": {
            "weighted_delta": g_wdelta,
            "verdict": g_weighted,
            "n_band_cells": len(band),
            "total_episodes": sum(c["n_total"] for c in band),
        },
        "per_axis": per_axis,
    }


def aggregate_action_cells(
    records: list[dict],
    task_classification: dict[str, dict],
) -> dict[CellKey, CellAction]:
    """Pool per-task rollout records into (condition, level) cells.

    Each record needs ``task_name`` plus successes/trials. Accepts the field names used by
    the existing camera eval: ``successes``/``trials`` (preferred) or ``n_success``/``n_total``.
    """
    cells: dict[CellKey, CellAction] = {}
    for rec in records:
        name = rec["task_name"]
        condition = _classify_condition(name)
        if condition is None:
            continue  # nominal-view task; not a perturbation cell
        meta = task_classification.get(name)
        if meta is None:
            continue
        level = int(meta["difficulty_level"])
        n_succ = int(rec.get("successes", rec.get("n_success", 0)))
        n_tot = int(rec.get("trials", rec.get("n_total", 0)))
        if n_tot <= 0:
            continue
        key = CellKey(condition, level)
        if key not in cells:
            cells[key] = CellAction(key, 0, 0)
        cells[key].n_success += n_succ
        cells[key].n_total += n_tot
    return cells


def evaluate_dissociation(
    action_cells: dict[CellKey, CellAction],
    excess_fvd_by_cell: dict[str, float],
    nominal_excess_fvd: Optional[float],
    config: DissociationConfig,
    semantic_preserved_by_cell: Optional[dict[str, bool]] = None,
    nominal_oracle: Optional[float] = None,
) -> DissociationResult:
    """Apply the LD13 dissociation criterion (FID-adapted). Pure logic (CPU-testable).

    Delta normalization: excess-FID can be <= 0 (the model can match GT as well as the
    GT-vs-GT floor), so LD13's ratio over excess(nom) is ill-posed. When the nominal oracle
    FID is supplied we use it (always > 0, the natural within-distribution FID scale) as the
    Delta denominator: Delta(c) = (excess(c) - excess(nom)) / (oracle(nom) + eps). This
    adaptation of the pre-registered formula is necessitated by the FVD->FID switch and is
    recorded for researcher confirmation; falling back to the raw LD13 ratio is unsafe.
    """
    result = DissociationResult(config=asdict(config), nominal_excess_fvd=nominal_excess_fvd)
    semantic_preserved_by_cell = semantic_preserved_by_cell or {}
    nominal_pp = config.nominal_success_rate * 100.0
    denom = (nominal_oracle if (nominal_oracle is not None and nominal_oracle > 0) else None)
    if denom is None and nominal_excess_fvd is not None:
        result.notes.append(
            "Delta denominator: nominal oracle FID not available; excess(nom)-normalized Delta "
            "is unreliable when excess(nom) is near 0 or negative. Supply --nominal_manifest so "
            "the FID stage records _nominal_oracle, or treat fidelity verdicts as provisional."
        )

    band_fidelity_flags: list[bool] = []
    for key in sorted(action_cells, key=lambda k: (k.condition, k.level)):
        cell = action_cells[key]
        kstr = str(key)
        drop_pp = nominal_pp - cell.success_rate * 100.0
        enough = cell.n_total >= config.min_episodes_per_cell
        in_band = enough and drop_pp >= config.action_drop_threshold_pp

        excess = excess_fvd_by_cell.get(kstr)
        rel_deg: Optional[float] = None
        fidelity_ok: Optional[bool] = None
        if excess is not None and nominal_excess_fvd is not None and denom is not None:
            rel_deg = (excess - nominal_excess_fvd) / (denom + config.eps)
            fidelity_ok = rel_deg <= config.rel_excess_fvd_threshold
        # If denom is None (no positive nominal oracle) we leave rel_deg/fidelity unset rather
        # than fabricate a verdict from an ill-posed ratio; the note above explains.

        sem_ok = semantic_preserved_by_cell.get(kstr)

        verdict = CellVerdict(
            key=kstr,
            n_total=cell.n_total,
            action_success_rate=cell.success_rate,
            action_drop_pp=drop_pp,
            in_severity_band=in_band,
            excess_fvd=excess,
            rel_excess_fvd_degradation=rel_deg,
            fidelity_preserved=fidelity_ok,
            semantic_preserved=sem_ok,
        )
        result.cells.append(asdict(verdict))
        if not enough:
            result.underpowered_cells.append(kstr)
        if in_band:
            result.severity_band_cells.append(kstr)
            if fidelity_ok is not None:
                # Semantic, when present, is ANDed with fidelity (LD13: both must survive).
                cell_holds = fidelity_ok and (sem_ok is not False)
                band_fidelity_flags.append(cell_holds)

    if not result.severity_band_cells:
        result.notes.append(
            "No aggregated cell reached the >=20pp action-drop severity band with >=100 "
            "episodes; cannot evaluate the dissociation criterion yet (need more episodes "
            "or stronger perturbation cells)."
        )
        result.dissociation_holds = None
    elif not band_fidelity_flags:
        result.notes.append(
            "Severity-band cells exist but their excess-FVD is missing; run the video-side "
            "measurement (e1_main_fvd) to complete the criterion."
        )
        result.dissociation_holds = None
    else:
        result.dissociation_holds = all(band_fidelity_flags)

    result.semantic_status = "missing" if not semantic_preserved_by_cell else "provisional"
    if semantic_preserved_by_cell == {}:
        result.notes.append(
            "Semantic-correctness readout (LD13c) NOT supplied — verdict uses fidelity only. "
            "The full LD13 claim additionally requires semantic trajectory correctness to "
            "survive; see handoff E1-main 'open design decision'."
        )

    # Surface all three pre-registration readings; NONE is auto-promoted to the primary
    # criterion (researcher-only decision). dissociation_holds stays = global_strict for
    # backward compatibility.
    _compute_multi_verdicts(result, config.rel_excess_fvd_threshold)
    return result


def _fmt(v, p="") -> str:
    return "—" if v is None else (f"{v:{p}}" if p else str(v))


def format_summary_md(result: DissociationResult) -> str:
    fmt = _fmt
    lines = ["# E1-main dissociation readout (LOCKED DECISION 13)", ""]
    holds = result.dissociation_holds
    verdict = {True: "HOLDS", False: "FAILS", None: "INCONCLUSIVE"}[holds]
    lines.append(f"**Dissociation verdict: {verdict}**")
    lines.append("")
    cfg = result.config
    lines.append(
        f"- nominal action SR: {cfg['nominal_success_rate'] * 100:.1f}%  | "
        f"drop band >= {cfg['action_drop_threshold_pp']:.0f} pp  | "
        f"rel excess-FID threshold <= {cfg['rel_excess_fvd_threshold'] * 100:.0f}%  | "
        f"min episodes/cell {cfg['min_episodes_per_cell']}"
    )
    lines.append(f"- nominal excess-FID: {result.nominal_excess_fvd}")
    lines.append("")
    lines.append("| cell | episodes | action SR | drop (pp) | band | excess-FID | Δ(c) | fidelity ok | semantic ok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in result.cells:
        lines.append(
            f"| {c['key']} | {c['n_total']} | {c['action_success_rate'] * 100:.1f}% | "
            f"{c['action_drop_pp']:.1f} | {'Y' if c['in_severity_band'] else ''} | "
            f"{fmt(c['excess_fvd'], '.2f')} | "
            f"{fmt(c['rel_excess_fvd_degradation'], '.3f')} | "
            f"{fmt(c['fidelity_preserved'])} | {fmt(c['semantic_preserved'])} |"
        )
    lines.append("")
    if result.severity_band_cells:
        lines.append(f"Severity-band cells: {', '.join(result.severity_band_cells)}")
    if result.underpowered_cells:
        lines.append(f"Under-powered cells (<min episodes): {', '.join(result.underpowered_cells)}")
    lines.append("")

    # --- Multiple LD13 readings (no primary auto-selected) ---
    def vfmt(v) -> str:
        return {True: "PASS", False: "FAIL", None: "N/A"}[v]

    v = result.verdicts
    if v:
        lines.append("## Verdict readings (NO primary auto-selected — researcher decides)")
        lines.append("")
        gs = v["global_strict_all_cells"]
        gw = v["global_weighted_severity_band"]
        lines.append(
            f"- **global_strict_all_cells** (one-cell veto): **{vfmt(gs['verdict'])}**"
            + (f"  — fail cells: {', '.join(gs['fail_cells'])}" if gs["fail_cells"] else "")
        )
        wd = gw["weighted_delta"]
        lines.append(
            f"- **global_weighted_severity_band** (episode-weighted Δ): "
            f"Δ_w={fmt(wd, '.3f')} → **{vfmt(gw['verdict'])}**"
        )
        lines.append("- **per_axis**:")
        for axis in ("C1", "C2", "C3"):
            a = v["per_axis"][axis]
            lines.append(
                f"  - {axis} ({a['label']}): strict **{vfmt(a['strict_verdict'])}** / "
                f"weighted Δ_w={fmt(a['weighted_delta'], '.3f')} **{vfmt(a['weighted_verdict'])}** "
                f"({a['total_episodes']} ep, {a['n_band_cells']} band cells"
                + (f"; strict-fail: {', '.join(a['fail_cells_strict'])}" if a["fail_cells_strict"] else "")
                + ")"
            )
        lines.append("")
        lines.append(
            f"> semantic_status: **{result.semantic_status}** — all readings above are "
            f"FIDELITY-SIDE ONLY (excess-FID). Full LD13 (fidelity AND LD13c semantic "
            f"correctness) is NOT verified. Do not write 'LD13 verified'."
        )
        lines.append("")

    for note in result.notes:
        lines.append(f"> NOTE: {note}")
    lines.append("")
    lines.append(f"Framing-matrix readout (recorded only, researcher decides): {result.framing_matrix_readout}")
    return "\n".join(lines)


def write_report(result: DissociationResult, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "e1_main_dissociation.json"
    md_path = out / "summary.md"
    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    md_path.write_text(format_summary_md(result), encoding="utf-8")
    return {"json": str(json_path), "summary": str(md_path)}


def _load_records(jsonl_files: list[str]) -> list[dict]:
    return parse_jsonl_files(jsonl_files)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="E1-main dissociation criterion + report")
    ap.add_argument("--jsonl_files", nargs="+", required=True, help="per-task rollout result JSONL(s)")
    ap.add_argument("--task_classification", required=True, help="task_classification.json")
    ap.add_argument(
        "--excess_fvd_json",
        default="",
        help="JSON of per-cell camera-conditioned excess-FID (LEGACY arg name: the metric is "
        "excess-FID, not FVD — Cosmos emits a single future frame/query). Keys: cell -> excess-FID; "
        "'_nominal' = nominal-bin excess-FID; '_nominal_oracle' = positive FID floor (Δ denominator); "
        "'_measured_tasks' = video-side task universe.",
    )
    ap.add_argument("--nominal_success_rate", type=float, required=True, help="unperturbed action SR in [0,1]")
    ap.add_argument("--semantic_json", default="", help="JSON: {cell_key: bool} semantic-preserved (optional)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    records = _load_records(args.jsonl_files)
    task_cls = load_task_classification(args.task_classification)

    excess_by_cell: dict[str, float] = {}
    nominal_excess = None
    nominal_oracle = None
    measured_tasks = None
    if args.excess_fvd_json:
        raw = json.loads(Path(args.excess_fvd_json).read_text(encoding="utf-8"))
        nominal_excess = raw.pop("_nominal", None)
        nominal_oracle = raw.pop("_nominal_oracle", None)  # positive FID-floor scale for Delta
        measured_tasks = raw.pop("_measured_tasks", None)  # video-side task universe (subset-safe)
        excess_by_cell = {str(k): float(v) for k, v in raw.items()}

    coverage_note = None
    if measured_tasks:
        # Restrict the action side to the SAME tasks the video side measured, so a subset/smoke
        # run does not pair full-universe action SR against subset video FID (the two sides must
        # cover the same tasks for the per-cell dissociation comparison to be meaningful).
        measured = set(measured_tasks)
        n_before = len({r.get("task_name") for r in records})
        records = [r for r in records if r.get("task_name") in measured]
        n_after = len({r.get("task_name") for r in records})
        if n_after < n_before:
            coverage_note = (
                f"PARTIAL/SMOKE COVERAGE: action side restricted to the {n_after} task(s) the "
                f"video side measured (of {n_before} in the action JSONL). This is expected for "
                f"subset runs; the dissociation verdict is NOT authoritative until the full "
                f"camera track is measured on both sides."
            )

    action_cells = aggregate_action_cells(records, task_cls)

    semantic_by_cell = None
    if args.semantic_json:
        semantic_by_cell = {str(k): bool(v) for k, v in json.loads(Path(args.semantic_json).read_text()).items()}

    cfg = DissociationConfig(nominal_success_rate=args.nominal_success_rate)
    result = evaluate_dissociation(
        action_cells, excess_by_cell, nominal_excess, cfg, semantic_by_cell, nominal_oracle=nominal_oracle
    )
    if coverage_note:
        result.notes.insert(0, coverage_note)
    paths = write_report(result, args.out_dir)
    print(f"E1-main report written: {paths['summary']}")


if __name__ == "__main__":
    main()
