#!/usr/bin/env python3
"""Aggregate results across worlds, governance regimes, and model families."""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from token_fed.models import REGIME_LABELS, REGIMES, WORLD_LABELS, WORLDS

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _mean_ci(vals: list[float], seed: int = 42) -> dict[str, Any]:
    if not vals:
        return {"mean": 0.0, "sd": 0.0, "ci95": [0.0, 0.0]}
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    if len(vals) == 1:
        return {"mean": round(m, 3), "sd": 0.0, "ci95": [round(m, 3), round(m, 3)]}
    rng = random.Random(seed)
    boots = sorted(
        sum(rng.choice(vals) for _ in range(len(vals))) / len(vals)
        for _ in range(2000)
    )
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return {"mean": round(m, 3), "sd": round(sd, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def _agg_group(eps: list[dict[str, Any]]) -> dict[str, Any]:
    if not eps:
        return {}
    keys = (
        "net_total_value",
        "gross_total_value",
        "external_real_value",
        "max_external_value",
        "training_real_value",
        "fiscal_deficit_penalty",
        "public_fiscal_margin",
        "high_impact_on_time_pct",
        "science_on_time_pct",
        "startup_on_time_pct",
        "enterprise_on_time_pct",
        "swarm_on_time_pct",
        "builder_tested_pct",
        "broken_prod_total",
        "cluster_utilization_pct",
        "avg_training_tokens",
        "total_borrowed_credits",
        "unpaid_loan_defaults",
    )
    out: dict[str, Any] = {"n_episodes": len(eps)}
    for k in keys:
        vals = [float(ep["summary"][k]) for ep in eps]
        out[k] = _mean_ci(vals)
    return out


def analyze(in_path: Path) -> dict[str, Any]:
    raw = json.loads(in_path.read_text())
    episodes: list[dict[str, Any]] = raw["episodes"]

    overall_by_regime = {
        reg: {
            "label": REGIME_LABELS[reg],
            **_agg_group([e for e in episodes if e["summary"]["regime"] == reg]),
        }
        for reg in REGIMES
    }

    by_world: dict[str, Any] = {}
    for w in WORLDS:
        w_eps = [e for e in episodes if e["summary"]["world"] == w]
        if not w_eps:
            continue
        by_world[w] = {
            "label": WORLD_LABELS[w],
            "by_regime": {
                reg: {
                    "label": REGIME_LABELS[reg],
                    **_agg_group([e for e in w_eps if e["summary"]["regime"] == reg]),
                }
                for reg in REGIMES
            },
        }

    models_present = sorted({e["summary"]["model_family"] for e in episodes})
    by_model: dict[str, Any] = {}
    for mf in models_present:
        m_eps = [e for e in episodes if e["summary"]["model_family"] == mf]
        by_model[mf] = {
            reg: _agg_group([e for e in m_eps if e["summary"]["regime"] == reg])
            for reg in REGIMES
        }

    # Paired head-to-head comparisons matched by `(model_family, world, seed)`
    indexed = {
        (e["summary"]["model_family"], e["summary"]["world"], int(e["summary"]["seed"]), e["summary"]["regime"]): e["summary"]
        for e in episodes
    }
    pair_specs = [
        ("llm_fed", "flat_today"),
        ("llm_fed", "pure_market"),
        ("llm_fed", "rule_fed"),
        ("rule_fed", "pure_market"),
        ("pure_market", "flat_today"),
    ]
    paired: dict[str, Any] = {}
    base_keys = sorted({(k[0], k[1], k[2]) for k in indexed})
    for a, b in pair_specs:
        diffs_val: list[float] = []
        diffs_hi: list[float] = []
        diffs_sci: list[float] = []
        wins = 0
        ties = 0
        for mf, w, sd in base_keys:
            sa = indexed.get((mf, w, sd, a))
            sb = indexed.get((mf, w, sd, b))
            if not sa or not sb:
                continue
            dv = float(sa["net_total_value"]) - float(sb["net_total_value"])
            diffs_val.append(dv)
            diffs_hi.append(float(sa["high_impact_on_time_pct"]) - float(sb["high_impact_on_time_pct"]))
            diffs_sci.append(float(sa["science_on_time_pct"]) - float(sb["science_on_time_pct"]))
            if dv > 1e-6:
                wins += 1
            elif abs(dv) <= 1e-6:
                ties += 1
        paired[f"{a}_vs_{b}"] = {
            "n_pairs": len(diffs_val),
            "wins": wins,
            "ties": ties,
            "net_value_diff": _mean_ci(diffs_val),
            "high_impact_pct_diff": _mean_ci(diffs_hi),
            "science_pct_diff": _mean_ci(diffs_sci),
        }

    # Extract representative Round-2 & Round-3 FOMC statements from `llm_fed`
    fomc_samples: dict[str, list[dict[str, Any]]] = {}
    for w in WORLDS:
        llm_eps = [e for e in episodes if e["summary"]["world"] == w and e["summary"]["regime"] == "llm_fed"]
        if llm_eps:
            ep0 = llm_eps[0]
            fomc_samples[w] = [
                {
                    "round": rl["round"],
                    "training_tokens": rl["training_tokens"],
                    "clearing_spot": rl["clearing_spot"],
                    "science_mult": rl["science_mult"],
                    "science_cap_tokens": rl["science_cap_tokens"],
                    "startup_credit_line": rl["startup_credit_line"],
                    "statement": rl["fomc_statement"],
                }
                for rl in ep0["rounds"]
            ]

    summary_obj = {
        "source_file": in_path.name,
        "n_total_episodes": len(episodes),
        "models": models_present,
        "runner_telemetry": raw.get("runner_telemetry", {}),
        "overall_by_regime": overall_by_regime,
        "by_world": by_world,
        "by_model": by_model,
        "paired_comparisons": paired,
        "fomc_samples": fomc_samples,
    }

    out_json = RESULTS_DIR / "summary.json"
    out_json.write_text(json.dumps(summary_obj, indent=2))

    # Render clean Markdown summary
    clean_names = {
        "flat_today": "1. Today (Flat + Locked Blocks)",
        "pure_market": "2. Pure Market (Auction + Resell)",
        "rule_fed": "3. Simple Fed Rule (Fixed Formula)",
        "llm_fed": "4. AI TokenFed (Live Central Banker)",
    }
    clean_worlds = {
        "supply_shock": "GPU Crunch",
        "slop_bubble": "Spam Bubble",
        "innovation_boom": "Science Boom",
    }
    lines = [
        f"# TokenFed — Full Benchmark Tables ({len(episodes)} Runs · 2,304 Coding Jobs)",
        "",
        "*Percentages show **% of each group's jobs finished on time**. Overall Score = customer output + lab R&D.*",
        "",
        "## All Scenarios Combined",
        "",
        "| System | Overall Score | Urgent Done | Science Done | Startups Done | Enterprise Done | Rich Swarms Done | Lab R&D | Public Receipts |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for reg in REGIMES:
        g = overall_by_regime[reg]
        if not g.get("n_episodes"):
            continue
        lines.append(
            f"| {clean_names[reg]} | **{g['net_total_value']['mean']:.1f}** | "
            f"{g['high_impact_on_time_pct']['mean']:.0f}% | "
            f"{g['science_on_time_pct']['mean']:.0f}% | "
            f"{g['startup_on_time_pct']['mean']:.0f}% | "
            f"{g['enterprise_on_time_pct']['mean']:.0f}% | "
            f"{g['swarm_on_time_pct']['mean']:.0f}% | "
            f"{g['training_real_value']['mean']:.1f} | "
            f"{g['public_fiscal_margin']['mean']:.2f}x |"
        )

    for w, w_data in by_world.items():
        lines.extend([
            "",
            f"## Scenario: {clean_worlds.get(w, w_data['label'])}",
            "",
            "| System | Overall Score | Urgent Done | Science Done | Startups Done | Enterprise Done | Rich Swarms Done | Avg Train Toks | Public Receipts |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for reg in REGIMES:
            g = w_data["by_regime"][reg]
            if not g.get("n_episodes"):
                continue
            lines.append(
                f"| {clean_names[reg]} | **{g['net_total_value']['mean']:.1f}** | "
                f"{g['high_impact_on_time_pct']['mean']:.0f}% | "
                f"{g['science_on_time_pct']['mean']:.0f}% | "
                f"{g['startup_on_time_pct']['mean']:.0f}% | "
                f"{g['enterprise_on_time_pct']['mean']:.0f}% | "
                f"{g['swarm_on_time_pct']['mean']:.0f}% | "
                f"{g['avg_training_tokens']['mean']:.0f} | "
                f"{g['public_fiscal_margin']['mean']:.2f}x |"
            )

    pair_labels = {
        "llm_fed_vs_flat_today": "AI TokenFed vs. Today",
        "llm_fed_vs_pure_market": "AI TokenFed vs. Pure Market",
        "llm_fed_vs_rule_fed": "AI TokenFed vs. Simple Fed Rule",
        "rule_fed_vs_pure_market": "Simple Fed Rule vs. Pure Market",
        "pure_market_vs_flat_today": "Pure Market vs. Today",
    }
    lines.extend(["", "## Paired Head-to-Head Comparisons (12 Matched Pairs Each)", ""])
    for pk, pv in paired.items():
        if pv["n_pairs"] == 0:
            continue
        nv = pv["net_value_diff"]
        hi = pv["high_impact_pct_diff"]
        sc = pv["science_pct_diff"]
        lines.append(
            f"- **{pair_labels.get(pk, pk)}**: Overall Score `{nv['mean']:+.1f}` "
            f"(95% CI `{nv['ci95'][0]:+.1f}..{nv['ci95'][1]:+.1f}`), "
            f"Urgent Done `{hi['mean']:+.1f}` pp, Science Done `{sc['mean']:+.1f}` pp, "
            f"Wins `{pv['wins']}/{pv['n_pairs']}`"
        )

    out_md = RESULTS_DIR / "summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json} and {out_md}")
    return summary_obj


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else (RESULTS_DIR / "benchmark_runs.json")
    analyze(target)
