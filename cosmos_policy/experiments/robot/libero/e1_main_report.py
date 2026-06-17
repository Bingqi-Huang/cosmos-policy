"""E1-main dissociation criterion + report (paper_outline LOCKED DECISION 13).

Combines the two primary readouts into the pre-registered dissociation judgment:

  * action side  : rollout success rate under perturbed cameras, stratified into cells
                   (condition C1/C2/C3 x difficulty level 1-5), reusing the existing
                   camera-eval classification so the cells match generate_camera_report.
  * video side   : camera-conditioned excess-FVD per cell + the relative degradation
                   Delta(c) vs the nominal bin (from e1_main_fvd).

Pre-registered criterion (LD13): over the aggregated severity band where action success
drops by >= 20 pp relative to nominal (cells with >= 100 episodes only), the relative
excess-FVD degradation must be <= 25%. Dissociation HOLDS when, in that band, fidelity
(and, when available, semantic correctness) survives while action success collapses.

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
    dissociation_holds: Optional[bool] = None
    framing_matrix_readout: str = "pending"  # recorded only; researcher decides the framing
    notes: list[str] = field(default_factory=list)


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
) -> DissociationResult:
    """Apply the LD13 pre-registered dissociation criterion. Pure logic (CPU-testable)."""
    result = DissociationResult(config=asdict(config), nominal_excess_fvd=nominal_excess_fvd)
    semantic_preserved_by_cell = semantic_preserved_by_cell or {}
    nominal_pp = config.nominal_success_rate * 100.0

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
        if excess is not None and nominal_excess_fvd is not None:
            rel_deg = (excess - nominal_excess_fvd) / (nominal_excess_fvd + config.eps)
            fidelity_ok = rel_deg <= config.rel_excess_fvd_threshold

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

    if semantic_preserved_by_cell == {}:
        result.notes.append(
            "Semantic-correctness readout (LD13c) NOT supplied — verdict uses fidelity only. "
            "The full LD13 claim additionally requires semantic trajectory correctness to "
            "survive; see handoff E1-main 'open design decision'."
        )
    return result


def format_summary_md(result: DissociationResult) -> str:
    lines = ["# E1-main dissociation readout (LOCKED DECISION 13)", ""]
    holds = result.dissociation_holds
    verdict = {True: "HOLDS", False: "FAILS", None: "INCONCLUSIVE"}[holds]
    lines.append(f"**Dissociation verdict: {verdict}**")
    lines.append("")
    cfg = result.config
    lines.append(
        f"- nominal action SR: {cfg['nominal_success_rate'] * 100:.1f}%  | "
        f"drop band >= {cfg['action_drop_threshold_pp']:.0f} pp  | "
        f"rel excess-FVD threshold <= {cfg['rel_excess_fvd_threshold'] * 100:.0f}%  | "
        f"min episodes/cell {cfg['min_episodes_per_cell']}"
    )
    lines.append(f"- nominal excess-FVD: {result.nominal_excess_fvd}")
    lines.append("")
    lines.append("| cell | episodes | action SR | drop (pp) | band | excess-FVD | Δ(c) | fidelity ok | semantic ok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in result.cells:
        def fmt(v, p=""):
            return "—" if v is None else (f"{v:{p}}" if p else str(v))

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
    ap.add_argument("--excess_fvd_json", default="", help="JSON: {cell_key: excess_fvd}; '_nominal' = nominal bin")
    ap.add_argument("--nominal_success_rate", type=float, required=True, help="unperturbed action SR in [0,1]")
    ap.add_argument("--semantic_json", default="", help="JSON: {cell_key: bool} semantic-preserved (optional)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    records = _load_records(args.jsonl_files)
    task_cls = load_task_classification(args.task_classification)
    action_cells = aggregate_action_cells(records, task_cls)

    excess_by_cell: dict[str, float] = {}
    nominal_excess = None
    if args.excess_fvd_json:
        raw = json.loads(Path(args.excess_fvd_json).read_text(encoding="utf-8"))
        nominal_excess = raw.pop("_nominal", None)
        excess_by_cell = {str(k): float(v) for k, v in raw.items()}

    semantic_by_cell = None
    if args.semantic_json:
        semantic_by_cell = {str(k): bool(v) for k, v in json.loads(Path(args.semantic_json).read_text()).items()}

    cfg = DissociationConfig(nominal_success_rate=args.nominal_success_rate)
    result = evaluate_dissociation(action_cells, excess_by_cell, nominal_excess, cfg, semantic_by_cell)
    paths = write_report(result, args.out_dir)
    print(f"E1-main report written: {paths['summary']}")


if __name__ == "__main__":
    main()
