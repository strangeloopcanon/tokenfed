#!/usr/bin/env python3
"""Export compact JSON payload for `docs/index.html` and chart scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from analyze import analyze
from token_fed.models import REGIMES, WORLDS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def export_dashboard(in_path: Path) -> Path:
    summary = analyze(in_path)

    def _pack_regime_map(rmap: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for reg in REGIMES:
            g = rmap.get(reg, {})
            if not g.get("n_episodes"):
                continue
            out[reg] = {
                "netVal": round(g["net_total_value"]["mean"], 1),
                "extVal": round(g["external_real_value"]["mean"], 1),
                "trnVal": round(g["training_real_value"]["mean"], 1),
                "defPen": round(g["fiscal_deficit_penalty"]["mean"], 2),
                "hiPct": round(g["high_impact_on_time_pct"]["mean"]),
                "sciPct": round(g["science_on_time_pct"]["mean"]),
                "strtPct": round(g["startup_on_time_pct"]["mean"]),
                "entPct": round(g["enterprise_on_time_pct"]["mean"]),
                "swmPct": round(g["swarm_on_time_pct"]["mean"]),
                "testPct": round(g["builder_tested_pct"]["mean"]),
                "utilPct": round(g["cluster_utilization_pct"]["mean"]),
                "trainTok": round(g["avg_training_tokens"]["mean"]),
                "borrowed": round(g["total_borrowed_credits"]["mean"]),
                "margin": f"{g['public_fiscal_margin']['mean']:.2f}x",
                "marginNum": round(g["public_fiscal_margin"]["mean"], 2),
            }
        return out

    payload = {
        "nEpisodes": summary["n_total_episodes"],
        "models": summary["models"],
        "telemetry": summary["runner_telemetry"],
        "overall": _pack_regime_map(summary["overall_by_regime"]),
        "worlds": {
            w: {
                "label": summary["by_world"][w]["label"],
                "regimes": _pack_regime_map(summary["by_world"][w]["by_regime"]),
            }
            for w in WORLDS
            if w in summary["by_world"]
        },
        "byModel": {
            mf: _pack_regime_map(summary["by_model"][mf])
            for mf in summary["models"]
        },
        "paired": summary["paired_comparisons"],
        "fomc": summary["fomc_samples"],
    }

    out_path = RESULTS_DIR / "dashboard_data.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (RESULTS_DIR / "benchmark_runs.json")
    export_dashboard(target)
