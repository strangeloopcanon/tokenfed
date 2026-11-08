#!/usr/bin/env python3
"""Benchmark runner & invariant self-test suite for the Token Fed multi-sector economy."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from token_fed.agy_client import MODEL_FAMILIES, AgyRunner
from token_fed.governor import MacroBriefing, build_market_board, decide_policy
from token_fed.models import REGIMES, WORLDS
from token_fed.scenarios import (
    WORLD_PHYSICAL_CAPACITY,
    build_world_agents,
    compute_training_value,
    training_bounds,
)
from token_fed.sim import _affordable_max_mult, run_episode
from token_fed.tasks import PROBLEMS, evaluate_code

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_self_tests() -> dict[str, Any]:
    """Deterministic unit & accounting invariants check before running live `agy` calls."""
    checks: list[str] = []

    # 1. Verify all 16 Python tasks pass both staging and held-out prod with reference solutions
    assert len(PROBLEMS) == 16, f"Expected 16 problems, got {len(PROBLEMS)}"
    for pid, prob in PROBLEMS.items():
        st = evaluate_code(prob.reference_solution, prob, suite="staging")
        pr = evaluate_code(prob.reference_solution, prob, suite="prod")
        assert st["all_passed"], f"Reference failed staging on {pid}: {st}"
        assert pr["all_passed"], f"Reference failed prod on {pid}: {pr}"
    checks.append("all_16_tasks_reference_verified")

    # 2. Verify concave internal training value schedule ordering
    v_1400 = compute_training_value(1400)
    v_2500 = compute_training_value(2500)
    v_3600 = compute_training_value(3600)
    v_4700 = compute_training_value(4700)
    assert 0.0 < v_1400 < v_2500 < v_3600 < v_4700
    m1 = (v_1400 - 0.0) / 1.4
    m2 = (v_3600 - v_1400) / 2.2
    m3 = (v_4700 - v_3600) / 1.1
    assert m1 > m2 > m3 > 0.0, f"Training return must be strictly concave: {(m1, m2, m3)}"
    assert 0.90 < m2 < 1.10, "Core training marginal value must sit between Swarm (0.45) and Builders (2.8+)"
    checks.append("concave_training_schedule_verified")

    # 3. Verify paired-seed world agent generation & Cash != Value structure across all 3 worlds
    for w in WORLDS:
        ags_a = build_world_agents(w, 101)
        ags_b = build_world_agents(w, 101)
        assert [a.agent_id for a in ags_a] == [a.agent_id for a in ags_b]
        assert len(ags_a) == 12
        sectors = {s: [a for a in ags_a if a.sector == s] for s in ("enterprise", "startup", "science", "swarm")}
        for s, lst in sectors.items():
            assert len(lst) == 3, f"Expected 3 agents in {s}"
        # Cash != Value check: Swarms have >3.5x Science cash wallet, while Science tasks have >8x Swarm real_value
        assert sectors["swarm"][0].wallet_credits > 3.5 * sectors["science"][0].wallet_credits
        assert sectors["science"][0].all_tickets[0].real_value > 8.0 * sectors["swarm"][0].all_tickets[0].real_value
    checks.append("paired_worlds_and_cash_neq_value_verified")

    # 4. Verify non-LLM policy bounds & slot conservation on MacroBriefing
    dummy_macro = MacroBriefing(
        round_num=2,
        total_rounds=5,
        physical_capacity_tokens=WORLD_PHYSICAL_CAPACITY["supply_shock"][2],
        total_reserved_ptu_tokens=3300,
        tier_summary={
            "enterprise": {"open_tickets": 3, "urgent_due_this_round": 2, "est_demand_tokens": 3450, "avg_wallet_credits": 3200.0},
            "verified_startup": {"open_tickets": 5, "urgent_due_this_round": 2, "est_demand_tokens": 5750, "avg_wallet_credits": 1100.0},
            "accredited_science": {"open_tickets": 3, "urgent_due_this_round": 0, "est_demand_tokens": 3450, "avg_wallet_credits": 1500.0},
            "general_commercial": {"open_tickets": 4, "urgent_due_this_round": 3, "est_demand_tokens": 4600, "avg_wallet_credits": 7000.0},
        },
    )
    for reg in ("flat_today", "pure_market", "rule_fed"):
        pol = asyncio.run(decide_policy(reg, dummy_macro))
        lo, hi = training_bounds(dummy_macro.physical_capacity_tokens)
        assert lo <= pol.training_tokens <= hi
        brd = build_market_board("supply_shock", pol, dummy_macro, released_ptu_tokens=1100 if pol.sublease_open else 0)
        kept = dummy_macro.total_reserved_ptu_tokens - brd.released_ptu_tokens
        assert pol.training_tokens + kept + brd.public_pool_tokens == dummy_macro.physical_capacity_tokens
    checks.append("policy_and_slot_conservation_verified")

    # 5. Verify binary-search affordability helper never exceeds wallet
    m_aff = _affordable_max_mult(wallet=1400.0, spill_draft=650, spill_ver=450, ver_mult_cap=0.75, desired_max_mult=3.0)
    cost = 650 * m_aff + 450 * min(m_aff, 0.75)
    assert cost <= 1400.0 + 1e-3 and cost >= 1390.0, f"Unexpected affordable mult {m_aff} -> cost {cost}"
    checks.append("affordability_clamp_verified")

    return {"passed": len(checks), "checks": checks}


async def run_suite(
    self_test_info: dict[str, Any],
    models: list[str],
    worlds: list[str],
    regimes: list[str],
    seeds: list[int],
    max_concurrency: int,
    episode_parallelism: int,
    out_path: Path,
    resume: bool,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing_episodes: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str, str, int]] = set()
    if resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        for ep in prev.get("episodes", []):
            s = ep["summary"]
            key = (s["model_family"], s["world"], s["regime"], int(s["seed"]))
            done_keys.add(key)
            existing_episodes.append(ep)
        print(f"[resume] Loaded {len(existing_episodes)} completed episodes from {out_path.name}")

    jobs: list[tuple[str, str, str, int]] = [
        (mf, w, reg, sd)
        for sd in seeds
        for mf in models
        for w in worlds
        for reg in regimes
        if (mf, w, reg, sd) not in done_keys
    ]
    total_target = len(existing_episodes) + len(jobs)
    print(
        f"[start] Running {len(jobs)} remaining episodes (target total={total_target}, "
        f"agy_concurrency={max_concurrency}, ep_parallel={episode_parallelism})"
    )

    runner = AgyRunner(max_concurrency=max_concurrency, timeout_sec=110.0)
    ep_sem = asyncio.Semaphore(episode_parallelism)
    save_lock = asyncio.Lock()
    t0 = time.monotonic()

    def _save_snapshot() -> None:
        payload = {
            "self_tests": self_test_info,
            "config": {
                "models": models,
                "worlds": worlds,
                "regimes": regimes,
                "seeds": seeds,
            },
            "runner_telemetry": {
                "total_agy_calls": runner.total_calls,
                "total_output_tokens": runner.total_output_tokens,
                "total_thinking_tokens": runner.total_thinking_tokens,
                "downgraded_calls": runner.downgraded_calls,
                "retry_events": runner.retry_events,
                "elapsed_wall_seconds": round(time.monotonic() - t0, 2),
            },
            "episodes": existing_episodes,
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out_path)

    async def _worker(mf: str, w: str, reg: str, sd: int) -> None:
        async with ep_sem:
            ep_t0 = time.monotonic()
            ep = await run_episode(runner, mf, w, reg, sd)
            dt = time.monotonic() - ep_t0
            s = ep["summary"]
            async with save_lock:
                existing_episodes.append(ep)
                _save_snapshot()
                print(
                    f"[{len(existing_episodes):02d}/{total_target:02d}] "
                    f"{mf:5s} | {w:15s} | {reg:11s} | seed={sd} | "
                    f"NetVal={s['net_total_value']:5.2f} (Ext={s['external_real_value']:5.2f}, Trn={s['training_real_value']:4.2f}, Pen={s['fiscal_deficit_penalty']:4.2f}) | "
                    f"Sci={s['science_on_time_pct']:5.1f}% Strt={s['startup_on_time_pct']:5.1f}% Ent={s['enterprise_on_time_pct']:5.1f}% Swm={s['swarm_on_time_pct']:5.1f}% | "
                    f"Margin={s['public_fiscal_margin']:.2f}x | {dt:.1f}s",
                    flush=True,
                )

    await asyncio.gather(*(_worker(mf, w, reg, sd) for mf, w, reg, sd in jobs))
    _save_snapshot()
    return json.loads(out_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Token Fed Economy Benchmark")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run 12-episode 3-world x 4-regime pilot (Argon, seed 101)")
    parser.add_argument("--models", nargs="+", default=["argon", "flash"], choices=list(MODEL_FAMILIES.keys()))
    parser.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--episode-parallelism", type=int, default=3)
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "benchmark_runs.json"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    self_test_info = run_self_tests()
    if args.self_test_only:
        print(json.dumps(self_test_info, indent=2))
        return

    if args.pilot:
        models = ["argon"]
        seeds = [101]
        out_path = RESULTS_DIR / "pilot_runs.json"
    else:
        models = args.models
        seeds = args.seeds
        out_path = Path(args.out)

    asyncio.run(
        run_suite(
            self_test_info=self_test_info,
            models=models,
            worlds=args.worlds,
            regimes=args.regimes,
            seeds=seeds,
            max_concurrency=args.max_concurrency,
            episode_parallelism=args.episode_parallelism,
            out_path=out_path,
            resume=not args.no_resume,
        )
    )


if __name__ == "__main__":
    main()
