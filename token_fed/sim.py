"""Multi-Sector Token Economy Simulation Engine (`flat_today`, `pure_market`, `rule_fed`, `llm_fed`).

Mechanical & Accounting Guarantees:
1. Physical Slot Conservation (`training + kept_ptu + public_pool == physical_capacity` every round).
2. Zero Omniscience (`rule_fed` and `llm_fed` never see scenario names, held-out tests, or `real_value`).
3. Steelmanned `pure_market` Comparator (uniform-price Spot auction + P2P PTU sublease + 0.5x Batch).
4. Closed Fiscal Balance Sheet (all Science/Verifier/Batch discounts & loan defaults must be covered
   by commercial surge margin + loan interest, or incur explicit fiscal deficit penalties below 0.95x).
5. 100% Live `agy` Execution (`0` output truncation, `0` held-out test leakage).
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict
from typing import Any

from token_fed.agy_client import (
    MODEL_FAMILIES,
    AgyRunner,
    extract_code_block,
    extract_json_object,
)
from token_fed.governor import (
    MacroBriefing,
    build_market_board,
    decide_policy,
    estimate_ptu_sublease_rate,
)
from token_fed.models import (
    FISCAL_DEFICIT_PENALTY_PER_1K,
    FISCAL_MARGIN_FLOOR,
    P2P_SUBLEASE_SHARE,
    PHYSICAL_TOKENS_429_OVERHEAD,
    PHYSICAL_TOKENS_BY_EFFORT,
    PHYSICAL_TOKENS_VERIFIER_PASS_ONLY,
    PHYSICAL_TOKENS_VERIFIER_RESERVE,
    REGIME_LABELS,
    WORKING_CAPITAL_INTEREST_RATE,
    WORLD_LABELS,
    AgentRoundPlan,
    AgentState,
    MarketBoard,
    TaskActionPlan,
    TaskTicket,
    TicketExecutionRecord,
)
from token_fed.scenarios import (
    TOTAL_ROUNDS,
    WORLD_PHYSICAL_CAPACITY,
    build_world_agents,
    compute_training_value,
)
from token_fed.tasks import PROBLEMS, CodingProblem, evaluate_code


def _max_liquid_cost_at_mult(
    draft_phys: int,
    ver_phys: int,
    own_ptu_avail: int,
    spot_mult: float,
    ver_mult_cap: float,
) -> tuple[float, int, int, int]:
    """Compute exact max pre-authorization liquid credit cost at `spot_mult`."""
    total_phys = draft_phys + ver_phys
    covered = min(own_ptu_avail, total_phys)
    spill = total_phys - covered
    if spill == 0 or total_phys == 0:
        return (0.0, covered, 0, 0)
    spill_draft = int(round(spill * (draft_phys / float(total_phys))))
    spill_ver = spill - spill_draft
    eff_ver_mult = min(spot_mult, ver_mult_cap)
    cost = round(spill_draft * spot_mult + spill_ver * eff_ver_mult, 2)
    return (cost, covered, spill, spill_ver)


def _affordable_max_mult(
    wallet: float,
    spill_draft: int,
    spill_ver: int,
    ver_mult_cap: float,
    desired_max_mult: float,
) -> float:
    """Clamp `desired_max_mult` to the highest Spot multiplier strictly affordable by `wallet`."""
    if spill_draft + spill_ver <= 0:
        return desired_max_mult
    lo = 0.0
    hi = max(0.0, float(desired_max_mult))
    # Binary search exact affordability with capped verifier rate
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        cost = spill_draft * mid + spill_ver * min(mid, ver_mult_cap)
        if cost <= wallet + 1e-6:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


async def _plan_ptu_stage1(
    runner: AgyRunner,
    planner_model: str,
    agent: AgentState,
    round_num: int,
    total_rounds: int,
    sublease_est_rate: float,
    prev_util: float,
    open_tickets: list[TaskTicket],
) -> tuple[int, dict[str, int]]:
    """Stage 1: Enterprise PTU holder decides how many idle PTU tokens to sublease."""
    if agent.ptu_reserved_per_round <= 0 or sublease_est_rate <= 0.0:
        return (0, {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0})

    open_summary = [
        {
            "ticket_id": tk.ticket_id,
            "priority": tk.priority,
            "deadline_round": tk.deadline_round,
            "cash_reward": tk.private_cash_reward,
        }
        for tk in open_tickets
    ]
    t_fast = PHYSICAL_TOKENS_BY_EFFORT["fast"]
    t_deep = PHYSICAL_TOKENS_BY_EFFORT["deep"]
    t_ver = PHYSICAL_TOKENS_VERIFIER_RESERVE

    prompt = f"""You are autonomous enterprise agent `{agent.agent_id}` in Round {round_num} of {total_rounds} (Stage 1: Voluntary PTU Sublease Exchange).
Role: {agent.role_desc}

SNAPSHOT:
- Reserved PTU Capacity This Round: {agent.ptu_reserved_per_round} slot-tokens (refreshes every round; unused PTUs expire at end of Round {round_num}. Task slot sizes: fast+verifier={t_fast + t_ver}, fast_only={t_fast}, deep+verifier={t_deep + t_ver}, deep_only={t_deep}).
- Liquid Wallet: {round(agent.wallet_credits, 1)} credits
- Open Tickets Now: {json.dumps(open_summary)}
- Estimated P2P Sublease Payout: ~{sublease_est_rate} credits per subleased PTU token (paid from 85% of realized public buyer spend at end of Round {round_num})
- Previous Public Pool Utilization: {round(prev_util, 2)}

Choose `release_ptu_tokens` (`0..{agent.ptu_reserved_per_round}`) to sublease into the public pool for Round {round_num}. Keep PTUs you need for urgent/own tasks this round; sublease idle PTUs to earn revenue.
Return ONLY JSON inside <JSON>...</JSON>:
<JSON>{{"release_ptu_tokens": 0, "rationale": "short reason"}}</JSON>"""

    res = await runner.complete(planner_model, prompt)
    usage = {
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
        "thinking_tokens": res["thinking_tokens"],
    }
    try:
        raw = extract_json_object(res["response"])
        rel = int(raw.get("release_ptu_tokens", 0) or 0)
    except (ValueError, TypeError):
        rel = agent.ptu_reserved_per_round if not open_tickets else 0
    rel = max(0, min(agent.ptu_reserved_per_round, rel))
    return (rel, usage)


def _build_agent_planner_prompt(
    agent: AgentState,
    board: MarketBoard,
    kept_ptu: int,
    open_tickets: list[TaskTicket],
) -> str:
    t_fast = PHYSICAL_TOKENS_BY_EFFORT["fast"]
    t_deep = PHYSICAL_TOKENS_BY_EFFORT["deep"]
    t_ver = PHYSICAL_TOKENS_VERIFIER_RESERVE
    t_ver_pass = PHYSICAL_TOKENS_VERIFIER_PASS_ONLY

    ticket_rows = [
        {
            "ticket_id": tk.ticket_id,
            "priority": tk.priority,
            "deadline_round": tk.deadline_round,
            "verified_on_time_cash_reward": tk.private_cash_reward,
            "broken_prod_cash_penalty": tk.private_broken_penalty,
            "complexity": "complex_legacy_module" if tk.problem_id in (
                "proration_invoice", "lease_fencing", "quota_rollover", "sla_waterfall",
                "wal_compactor", "dag_retry_engine", "sliding_limiter", "bitemporal_book",
            ) else "compact_component",
            "prior_status": agent.failed_attempts.get(tk.ticket_id, "fresh"),
        }
        for tk in open_tickets
    ]

    can_borrow = (
        agent.kyc_tier == "verified_startup"
        and board.startup_credit_line > agent.outstanding_debt
        and agent.wallet_credits < 1600
    )
    max_borrow = round(max(0.0, board.startup_credit_line - agent.outstanding_debt), 1) if can_borrow else 0.0

    sci_note = ""
    if agent.kyc_tier == "accredited_science" and board.science_cap_tokens > 0:
        sci_note = (
            f"\n- ACCREDITED SCIENCE PRIORITY WINDOW: Up to {board.science_cap_tokens} public tokens "
            f"ring-fenced at `{board.science_multiplier}x` credits/token for `spot` submissions "
            f"(set `max_spot_mult` >= {board.science_multiplier} to qualify, or higher to also bid in commercial overflow)!"
        )

    batch_note = (
        f"OPEN at `{board.batch_multiplier}x` credits/token (runs on round slack; delivers at Round {board.round_num + 1} — "
        f"so ONLY on-time for tickets with `deadline_round >= {board.round_num + 1}`! Unstarted Batch jobs auto-refund next round.)"
        if board.batch_available
        else "CLOSED (not available in this regime)"
    )

    credit_note = (
        f"AVAILABLE: You may borrow `0..{max_borrow}` working-capital credits (`borrow_credits`) at 5% interest, "
        "automatically repaid out of your earned delivery revenues."
        if max_borrow > 0
        else "0.0 (not applicable)"
    )

    auction_note = (
        "Fixed `1.00x` price; excess requests hit random HTTP 429s."
        if board.regime == "flat_today"
        else (
            f"Uniform-price Spot auction with floor `{board.spot_floor_multiplier}x` (prev clearing `{board.prev_clearing_spot}x`). "
            f"Specify `max_spot_mult` (your max willingness-to-pay per token >= floor, backed by your wallet) for each `spot` task."
        )
    )

    return f"""You are autonomous firm `{agent.agent_id}` (`{agent.kyc_tier}`) in Round {board.round_num} of {board.total_rounds}.
Role & Business Model: {agent.role_desc}

BALANCE SHEET:
- Liquid Wallet: {round(agent.wallet_credits, 1)} credits | Per-Round External Inflow Next Round: {agent.per_round_stipend} credits
- Outstanding Working-Capital Debt: {round(agent.outstanding_debt, 1)} credits
- Working-Capital Credit Facility This Round: {credit_note}
- Kept Own PTU Capacity This Round: {kept_ptu} tokens (0 liquid credits for tokens covered by own PTU)

ROUND {board.round_num} CLUSTER MARKET BOARD:
- Public Pool Supply: {board.public_pool_tokens} tokens | Prev Utilization: {round(board.prev_utilization, 2)} | Prev Priced-Out/429 Rate: {round(board.prev_throttle_or_priced_out_rate, 2)}
- Spot Clearing Rule: {auction_note}
- Staging Verifier Rate Cap: `{board.verifier_multiplier}x` credits/token (prev unverified bug rate: {round(board.prev_unverified_fail_rate, 2)}){sci_note}
- Async Batch Queue: {batch_note}

COMPUTE SIZES:
- `effort`: `"fast"` = {t_fast} draft tokens | `"deep"` = {t_deep} reasoning draft tokens (stronger on `complex_legacy_module`)
- `use_verifier`: `true` reserves +{t_ver} tokens for staging unit tests + bug repair (only burns +{t_ver_pass} and refunds {t_ver - t_ver_pass} if staging passes on 1st try!); `false` = +0 tokens (higher risk of shipping broken code and incurring broken penalties!).

OPEN TICKETS:
{json.dumps(ticket_rows, indent=2)}

Choose `borrow_credits` (`0..{max_borrow}`) and a plan for EVERY open ticket (`mode`: `"spot" | "batch" | "defer"`, `effort`: `"fast" | "deep"`, `use_verifier`: `true | false`, `max_spot_mult`: float).
Return ONLY JSON inside <JSON>...</JSON>:
<JSON>
{{
  "borrow_credits": 0,
  "tasks": [
    {{"ticket_id": "{open_tickets[0].ticket_id}", "mode": "spot", "effort": "fast", "use_verifier": true, "max_spot_mult": 1.5}}
  ],
  "rationale": "one short sentence"
}}
</JSON>"""


async def _plan_agent_round(
    runner: AgyRunner,
    planner_model: str,
    agent: AgentState,
    board: MarketBoard,
    kept_ptu: int,
    released_ptu: int,
    stage1_usage: dict[str, int],
    open_tickets: list[TaskTicket],
) -> AgentRoundPlan:
    if not open_tickets:
        return AgentRoundPlan(
            agent_id=agent.agent_id,
            round_num=board.round_num,
            release_ptu_tokens=released_ptu,
            rationale="no open tickets",
            planner_usage=stage1_usage,
        )

    prompt = _build_agent_planner_prompt(agent, board, kept_ptu, open_tickets)
    res = await runner.complete(planner_model, prompt)
    usage = {
        "input_tokens": stage1_usage.get("input_tokens", 0) + res["input_tokens"],
        "output_tokens": stage1_usage.get("output_tokens", 0) + res["output_tokens"],
        "thinking_tokens": stage1_usage.get("thinking_tokens", 0) + res["thinking_tokens"],
    }

    try:
        raw = extract_json_object(res["response"])
    except ValueError:
        raw = {}

    borrow_req = 0.0
    if agent.kyc_tier == "verified_startup" and board.startup_credit_line > agent.outstanding_debt:
        try:
            borrow_req = float(raw.get("borrow_credits", 0.0) or 0.0)
        except (TypeError, ValueError):
            borrow_req = 0.0
        borrow_req = max(0.0, min(board.startup_credit_line - agent.outstanding_debt, borrow_req))

    open_by_id = {tk.ticket_id: tk for tk in open_tickets}
    seen: set[str] = set()
    plans: list[TaskActionPlan] = []
    for item in raw.get("tasks", []) or []:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("ticket_id", ""))
        if tid not in open_by_id or tid in seen:
            continue
        seen.add(tid)
        mode = str(item.get("mode", "defer")).lower()
        if mode not in ("spot", "batch", "defer"):
            mode = "defer"
        if mode == "batch" and not board.batch_available:
            mode = "spot" if open_by_id[tid].deadline_round <= board.round_num else "defer"
        effort = str(item.get("effort", "fast")).lower()
        if effort not in PHYSICAL_TOKENS_BY_EFFORT:
            effort = "fast"
        use_ver = bool(item.get("use_verifier", True))
        try:
            max_mult = float(item.get("max_spot_mult", board.spot_floor_multiplier) or board.spot_floor_multiplier)
        except (TypeError, ValueError):
            max_mult = board.spot_floor_multiplier
        max_mult = max(0.30, min(4.00, max_mult))
        plans.append(
            TaskActionPlan(
                ticket_id=tid,
                mode=mode,
                effort=effort,
                use_verifier=use_ver,
                max_spot_mult=round(max_mult, 2),
            )
        )

    for tk in open_tickets:
        if tk.ticket_id not in seen:
            default_mode = "spot" if tk.deadline_round <= board.round_num else "defer"
            plans.append(
                TaskActionPlan(
                    ticket_id=tk.ticket_id,
                    mode=default_mode,
                    effort="fast",
                    use_verifier=True,
                    max_spot_mult=max(1.5, board.spot_floor_multiplier),
                )
            )

    return AgentRoundPlan(
        agent_id=agent.agent_id,
        round_num=board.round_num,
        release_ptu_tokens=released_ptu,
        borrow_credits=round(borrow_req, 2),
        task_plans=plans,
        rationale=str(raw.get("rationale", ""))[:200],
        planner_usage=usage,
    )


async def _execute_admitted_ticket(
    runner: AgyRunner,
    family_cfg: dict[str, str],
    ticket: TaskTicket,
    round_num: int,
    delivery_round: int,
    channel: str,
    effort: str,
    use_verifier: bool,
    prior_note: str,
) -> dict[str, Any]:
    """Run live `agy` Python generation + optional staging test/repair + held-out prod evaluation."""
    prob: CodingProblem = PROBLEMS[ticket.problem_id]
    slot_tokens = PHYSICAL_TOKENS_BY_EFFORT[effort]

    if effort == "deep":
        draft_model = family_cfg["deep"]
        style_note = (
            "Reason carefully through all invariants and edge cases, then write the complete Python "
            "implementation inside <CODE>...</CODE> with NO docstrings, NO comments, and zero prose."
        )
    else:
        draft_model = family_cfg["lean"]
        style_note = (
            "Write the complete Python implementation inside <CODE>...</CODE> with NO docstrings, "
            "NO comments, and zero prose."
        )

    retry_ctx = f"\nPrevious attempt status: {prior_note}\n" if prior_note else ""
    draft_prompt = f"{style_note}{retry_ctx}\nSpecification:\n{prob.spec}\n"

    draft_res = await runner.complete(draft_model, draft_prompt)
    candidate_code = extract_code_block(draft_res["response"])
    total_llm_in = draft_res["input_tokens"]
    total_llm_out = draft_res["output_tokens"]
    total_llm_think = draft_res["thinking_tokens"]

    staging_first_ok: bool | None = None
    repair_ran = False
    verifier_phys_burned = 0

    if use_verifier:
        st_eval = evaluate_code(candidate_code, prob, suite="staging")
        staging_first_ok = bool(st_eval["all_passed"])
        if staging_first_ok:
            verifier_phys_burned = PHYSICAL_TOKENS_VERIFIER_PASS_ONLY
        else:
            repair_ran = True
            verifier_phys_burned = PHYSICAL_TOKENS_VERIFIER_RESERVE
            fail_lines = "\n".join(st_eval["failures"][:4])
            repair_prompt = (
                "Staging verification caught failures. Output the complete corrected Python implementation "
                "inside <CODE>...</CODE> (NO docstrings, NO comments, zero prose).\n\n"
                f"Specification:\n{prob.spec}\n\n"
                f"Candidate code:\n```python\n{candidate_code}\n```\n\n"
                f"Staging failures:\n{fail_lines}\n"
            )
            rep_res = await runner.complete(family_cfg["repair"], repair_prompt)
            total_llm_in += rep_res["input_tokens"]
            total_llm_out += rep_res["output_tokens"]
            total_llm_think += rep_res["thinking_tokens"]
            candidate_code = extract_code_block(rep_res["response"])

    prod_eval = evaluate_code(candidate_code, prob, suite="prod")
    prod_ok = bool(prod_eval["all_passed"])
    err_summary = "" if prod_ok else f"Prod suite failed ({prod_eval['passed']}/{prod_eval['total']} passed)"

    return {
        "ticket": ticket,
        "round_num": round_num,
        "delivery_round": delivery_round,
        "channel": channel,
        "effort": effort,
        "use_verifier": use_verifier,
        "staging_first_ok": staging_first_ok,
        "repair_ran": repair_ran,
        "prod_ok": prod_ok,
        "prod_passed": int(prod_eval["passed"]),
        "prod_total": int(prod_eval["total"]),
        "draft_phys": slot_tokens,
        "verifier_phys_burned": verifier_phys_burned,
        "llm_in": total_llm_in,
        "llm_out": total_llm_out,
        "llm_think": total_llm_think,
        "err_summary": err_summary,
    }


def _repay_working_capital_debt(agent: AgentState) -> tuple[float, float]:
    """Automatically sweep available liquid credits to retire outstanding working-capital debt."""
    if agent.outstanding_debt <= 1e-6 or agent.wallet_credits <= 1e-6:
        return (0.0, 0.0)
    pay = round(min(agent.wallet_credits, agent.outstanding_debt), 2)
    # Separate interest component (~5/105 of balance) for transparent reporting
    interest_part = round(pay * (WORKING_CAPITAL_INTEREST_RATE / (1.0 + WORKING_CAPITAL_INTEREST_RATE)), 2)
    agent.wallet_credits = round(max(0.0, agent.wallet_credits - pay), 2)
    agent.outstanding_debt = round(max(0.0, agent.outstanding_debt - pay), 2)
    agent.total_debt_repaid = round(agent.total_debt_repaid + pay, 2)
    agent.total_interest_paid = round(agent.total_interest_paid + interest_part, 2)
    return (pay, interest_part)


async def run_episode(
    runner: AgyRunner,
    model_family: str,
    world: str,
    regime: str,
    seed: int,
) -> dict[str, Any]:
    """Execute one complete 5-round, 12-agent Token Economy episode."""
    family_cfg = MODEL_FAMILIES[model_family]
    agents = build_world_agents(world, seed)
    agents_by_id = {a.agent_id: a for a in agents}
    total_ptu_committed = sum(a.ptu_reserved_per_round for a in agents)

    prev_clearing_spot = 1.00
    prev_util = 0.72
    prev_drop_rate = 0.0
    prev_drops_by_tier: dict[str, int] = {
        "enterprise": 0,
        "verified_startup": 0,
        "accredited_science": 0,
        "general_commercial": 0,
    }
    prev_unverified_fail_rate = 0.16
    prev_idle_ptu_ratio = 0.50

    global_batch_queue: list[dict[str, Any]] = []
    execution_records: list[TicketExecutionRecord] = []
    round_logs: list[dict[str, Any]] = []

    cum_public_buyer_spend = 0.0
    cum_ptu_sublease_paid = 0.0
    cum_interest_collected = 0.0
    cum_public_phys_burned = 0
    cum_training_value = 0.0
    total_fed_llm_tokens = 0
    total_planner_llm_tokens = 0

    for r in range(1, TOTAL_ROUNDS + 1):
        phys_cap = WORLD_PHYSICAL_CAPACITY[world][r]

        # 0. External stipends / bubble cashflows & refund unstarted prior-round Batch jobs
        for ag in agents:
            inflow = (ag.per_round_stipend if r >= 2 else 0.0) + ag.bubble_round_bonus.get(r, 0.0)
            if inflow > 0:
                ag.wallet_credits = round(ag.wallet_credits + inflow, 2)

        if global_batch_queue:
            for bjob in global_batch_queue:
                tk = bjob["ticket"]
                ag = agents_by_id[bjob["agent_id"]]
                ag.wallet_credits = round(ag.wallet_credits + bjob["liquid_precharged"], 2)
                ag.queued_in_batch.pop(tk.ticket_id, None)
                ag.failed_attempts[tk.ticket_id] = (
                    f"Batch queue full in Round {bjob['queued_round']}; refunded in full."
                )
            global_batch_queue = []

        open_by_agent: dict[str, list[TaskTicket]] = {
            ag.agent_id: [
                tk
                for tk in ag.all_tickets
                if tk.arrival_round <= r
                and tk.ticket_id not in ag.completed_tickets
                and tk.ticket_id not in ag.queued_in_batch
            ]
            for ag in agents
        }

        # Build observable KYC-tier telemetry (strictly zero hidden real_value or world label!)
        tier_summary: dict[str, dict[str, float | int]] = {}
        for tier in ("enterprise", "verified_startup", "accredited_science", "general_commercial"):
            t_agents = [ag for ag in agents if ag.kyc_tier == tier]
            t_open = [tk for ag in t_agents for tk in open_by_agent[ag.agent_id]]
            tier_summary[tier] = {
                "open_tickets": len(t_open),
                "urgent_due_this_round": sum(1 for tk in t_open if tk.deadline_round <= r),
                "est_demand_tokens": len(t_open) * 1150,
                "avg_wallet_credits": round(
                    sum(ag.wallet_credits for ag in t_agents) / max(1, len(t_agents)), 1
                ),
                "illiquid_firms_below_1400": sum(1 for ag in t_agents if ag.wallet_credits < 1400),
            }

        cur_net_rev = cum_public_buyer_spend - cum_ptu_sublease_paid + cum_interest_collected
        cur_margin = (cur_net_rev / float(cum_public_phys_burned)) if cum_public_phys_burned > 0 else 1.05

        macro = MacroBriefing(
            round_num=r,
            total_rounds=TOTAL_ROUNDS,
            physical_capacity_tokens=phys_cap,
            total_reserved_ptu_tokens=total_ptu_committed,
            tier_summary=tier_summary,
            prev_clearing_spot=prev_clearing_spot,
            prev_utilization=prev_util,
            prev_drop_rate=prev_drop_rate,
            prev_drops_by_tier=dict(prev_drops_by_tier),
            prev_unverified_fail_rate=prev_unverified_fail_rate,
            prev_idle_ptu_ratio=prev_idle_ptu_ratio,
            cum_fiscal_margin=round(cur_margin, 3),
            cum_public_tokens_burned=cum_public_phys_burned,
        )

        # 1. Policy Decision (`flat_today`, `pure_market`, `rule_fed`, or live `llm_fed`)
        policy = await decide_policy(
            regime=regime,
            macro=macro,
            runner=runner,
            fed_model=family_cfg["fed"],
        )
        total_fed_llm_tokens += (
            int(policy.fed_llm_usage.get("input_tokens", 0))
            + int(policy.fed_llm_usage.get("output_tokens", 0))
            + int(policy.fed_llm_usage.get("thinking_tokens", 0))
        )
        round_train_value = compute_training_value(policy.training_tokens)
        cum_training_value += round_train_value

        # 2. Stage-1 Enterprise PTU Sublease Clearing
        est_sublease = estimate_ptu_sublease_rate(policy, macro)
        ptu_agents = [ag for ag in agents if ag.ptu_reserved_per_round > 0]
        stage1_results = await asyncio.gather(
            *(
                _plan_ptu_stage1(
                    runner,
                    family_cfg["planner"],
                    ag,
                    r,
                    TOTAL_ROUNDS,
                    est_sublease,
                    prev_util,
                    open_by_agent[ag.agent_id],
                )
                for ag in ptu_agents
            )
        )
        stage1_rel_by_id: dict[str, int] = {ag.agent_id: 0 for ag in agents}
        stage1_usage_by_id: dict[str, dict[str, int]] = {
            ag.agent_id: {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0} for ag in agents
        }
        released_ptu_total = 0
        for ag, (rel_ptu, s1_u) in zip(ptu_agents, stage1_results):
            stage1_rel_by_id[ag.agent_id] = rel_ptu
            stage1_usage_by_id[ag.agent_id] = s1_u
            released_ptu_total += rel_ptu

        # 3. Stage-2 MarketBoard & 12 Agent-Firm Portfolio/Bidding Planners
        board = build_market_board(world, policy, macro, released_ptu_tokens=released_ptu_total)

        plans = await asyncio.gather(
            *(
                _plan_agent_round(
                    runner,
                    family_cfg["planner"],
                    ag,
                    board,
                    ag.ptu_reserved_per_round - stage1_rel_by_id[ag.agent_id],
                    stage1_rel_by_id[ag.agent_id],
                    stage1_usage_by_id[ag.agent_id],
                    open_by_agent[ag.agent_id],
                )
                for ag in agents
            )
        )
        plans_by_id = {p.agent_id: p for p in plans}
        for p in plans:
            pu = p.planner_usage
            total_planner_llm_tokens += (
                int(pu.get("input_tokens", 0))
                + int(pu.get("output_tokens", 0))
                + int(pu.get("thinking_tokens", 0))
            )

        # 4. Working-Capital Draws & Task Pre-Authorization
        used_ptu_total = 0
        idle_hoarded_ptu_total = 0
        ptu_admitted_jobs: list[dict[str, Any]] = []
        live_spot_requests: list[dict[str, Any]] = []
        newly_queued_batch: list[dict[str, Any]] = []
        drops_by_tier_round: dict[str, int] = {
            "enterprise": 0,
            "verified_startup": 0,
            "accredited_science": 0,
            "general_commercial": 0,
        }
        round_borrowed_total = 0.0

        for ag in agents:
            plan = plans_by_id[ag.agent_id]
            if plan.borrow_credits > 0 and ag.kyc_tier == "verified_startup":
                draw = round(min(plan.borrow_credits, max(0.0, board.startup_credit_line - ag.outstanding_debt)), 2)
                if draw > 0:
                    ag.wallet_credits = round(ag.wallet_credits + draw, 2)
                    ag.outstanding_debt = round(ag.outstanding_debt + draw * (1.0 + WORKING_CAPITAL_INTEREST_RATE), 2)
                    ag.total_borrowed = round(ag.total_borrowed + draw, 2)
                    round_borrowed_total += draw

            own_ptu_left = ag.ptu_reserved_per_round - plan.release_ptu_tokens
            tk_map = {tk.ticket_id: tk for tk in open_by_agent[ag.agent_id]}
            ordered_plans = sorted(
                plan.task_plans,
                key=lambda tp: (tk_map[tp.ticket_id].deadline_round, tk_map[tp.ticket_id].priority),
            )

            for tp in ordered_plans:
                if tp.mode == "defer":
                    continue
                tk = tk_map[tp.ticket_id]
                draft_phys = PHYSICAL_TOKENS_BY_EFFORT[tp.effort]
                ver_phys = PHYSICAL_TOKENS_VERIFIER_RESERVE if tp.use_verifier else 0
                total_phys = draft_phys + ver_phys

                if tp.mode == "spot":
                    # Floor rate needed for eligibility
                    is_sci_elig = (ag.kyc_tier == "accredited_science" and board.science_cap_tokens > 0)
                    min_floor = board.science_multiplier if is_sci_elig else board.spot_floor_multiplier
                    ver_cap_for_floor = (
                        board.science_multiplier
                        if is_sci_elig
                        else (board.spot_floor_multiplier if regime == "pure_market" else policy.verifier_mult_cap)
                    )
                    floor_cost, covered_by_ptu, spill_to_spot, spill_ver = _max_liquid_cost_at_mult(
                        draft_phys, ver_phys, own_ptu_left, min_floor, ver_cap_for_floor
                    )
                    if spill_to_spot == 0:
                        own_ptu_left -= covered_by_ptu
                        used_ptu_total += covered_by_ptu
                        ptu_admitted_jobs.append({
                            "agent_id": ag.agent_id,
                            "ticket": tk,
                            "effort": tp.effort,
                            "use_verifier": tp.use_verifier,
                            "draft_phys": draft_phys,
                            "ver_phys": ver_phys,
                            "covered_by_ptu": covered_by_ptu,
                            "public_spot_phys": 0,
                            "spill_draft_phys": 0,
                            "spill_ver_phys": 0,
                            "liquid_precharged": 0.0,
                            "effective_max_mult": tp.max_spot_mult,
                            "clearing_mult": 0.0,
                            "ver_cap": ver_cap_for_floor,
                        })
                        continue

                    spill_draft = spill_to_spot - spill_ver
                    desired_bid = 1.00 if regime == "flat_today" else tp.max_spot_mult
                    ver_cap_comm = 99.0 if regime == "pure_market" else policy.verifier_mult_cap
                    eff_max_mult = _affordable_max_mult(
                        ag.wallet_credits,
                        spill_draft,
                        spill_ver,
                        min(min_floor, ver_cap_comm) if is_sci_elig else ver_cap_comm,
                        desired_bid,
                    )

                    if floor_cost > ag.wallet_credits + 1e-6 or eff_max_mult + 1e-6 < min_floor:
                        drops_by_tier_round[ag.kyc_tier] += 1
                        ag.failed_attempts[tk.ticket_id] = (
                            f"Priced out in Round {r} (floor {min_floor:.2f}x needed {floor_cost:.0f} credits; wallet {ag.wallet_credits:.0f})"
                        )
                        execution_records.append(
                            TicketExecutionRecord(
                                ticket_id=tk.ticket_id,
                                problem_id=tk.problem_id,
                                agent_id=ag.agent_id,
                                sector=ag.sector,
                                kyc_tier=ag.kyc_tier,
                                priority=tk.priority,
                                arrival_round=tk.arrival_round,
                                deadline_round=tk.deadline_round,
                                executed_round=r,
                                delivery_round=r,
                                channel="spot",
                                effort=tp.effort,
                                used_verifier=tp.use_verifier,
                                bid_max_mult=eff_max_mult,
                                clearing_mult_paid=0.0,
                                priced_out=True,
                                error_summary=f"Priced out at floor {min_floor:.2f}x",
                            )
                        )
                        continue

                    # Pre-hold maximum bid cost from wallet (excess above uniform clearing refunded immediately after auction!)
                    max_hold_ver_cap = board.science_multiplier if (is_sci_elig and eff_max_mult <= board.science_multiplier) else ver_cap_comm
                    hold_cost = round(
                        spill_draft * eff_max_mult + spill_ver * min(eff_max_mult, max_hold_ver_cap), 2
                    )
                    hold_cost = min(ag.wallet_credits, hold_cost)
                    own_ptu_left -= covered_by_ptu
                    used_ptu_total += covered_by_ptu
                    ag.wallet_credits = round(max(0.0, ag.wallet_credits - hold_cost), 2)

                    live_spot_requests.append({
                        "agent_id": ag.agent_id,
                        "ticket": tk,
                        "effort": tp.effort,
                        "use_verifier": tp.use_verifier,
                        "draft_phys": draft_phys,
                        "ver_phys": ver_phys,
                        "covered_by_ptu": covered_by_ptu,
                        "public_spot_phys": spill_to_spot,
                        "spill_draft_phys": spill_draft,
                        "spill_ver_phys": spill_ver,
                        "liquid_precharged": hold_cost,
                        "effective_max_mult": eff_max_mult,
                        "ver_cap_comm": ver_cap_comm,
                    })

                elif tp.mode == "batch" and board.batch_available:
                    batch_cost = round(total_phys * board.batch_multiplier, 2)
                    if batch_cost > ag.wallet_credits + 1e-6:
                        drops_by_tier_round[ag.kyc_tier] += 1
                        ag.failed_attempts[tk.ticket_id] = (
                            f"Unaffordable for Batch in Round {r} (needed {batch_cost:.0f}, wallet {ag.wallet_credits:.0f})"
                        )
                        execution_records.append(
                            TicketExecutionRecord(
                                ticket_id=tk.ticket_id,
                                problem_id=tk.problem_id,
                                agent_id=ag.agent_id,
                                sector=ag.sector,
                                kyc_tier=ag.kyc_tier,
                                priority=tk.priority,
                                arrival_round=tk.arrival_round,
                                deadline_round=tk.deadline_round,
                                executed_round=r,
                                delivery_round=r + 1,
                                channel="batch",
                                effort=tp.effort,
                                used_verifier=tp.use_verifier,
                                priced_out=True,
                                error_summary="Unaffordable for Batch",
                            )
                        )
                        continue
                    ag.wallet_credits = round(max(0.0, ag.wallet_credits - batch_cost), 2)
                    ag.queued_in_batch[tk.ticket_id] = tp
                    newly_queued_batch.append({
                        "agent_id": ag.agent_id,
                        "ticket": tk,
                        "effort": tp.effort,
                        "use_verifier": tp.use_verifier,
                        "draft_phys": draft_phys,
                        "ver_phys": ver_phys,
                        "phys_reserved": total_phys,
                        "liquid_precharged": batch_cost,
                        "clearing_mult": board.batch_multiplier,
                        "queued_round": r,
                    })

            if own_ptu_left > 0:
                idle_hoarded_ptu_total += own_ptu_left

        # 5. Physical Slot Conservation Check
        assert (
            policy.training_tokens + used_ptu_total + idle_hoarded_ptu_total + board.public_pool_tokens
            == phys_cap
        ), "Physical GPU slot conservation violated!"

        remaining_public = board.public_pool_tokens
        arrival_rng = random.Random(f"{world}:{seed}:round:{r}")

        science_admitted_jobs: list[dict[str, Any]] = []
        spot_admitted_jobs: list[dict[str, Any]] = []
        throttled_429_count = 0
        auction_outbid_count = 0
        public_burned_429 = 0
        round_public_buyer_spend = 0.0
        clearing_spot_realized = board.spot_floor_multiplier

        if regime == "flat_today":
            shuffled = list(live_spot_requests)
            arrival_rng.shuffle(shuffled)
            for req in shuffled:
                need = req["public_spot_phys"]
                if need <= remaining_public:
                    remaining_public -= need
                    req["clearing_mult"] = 1.00
                    req["ver_unit_mult"] = 1.00
                    spot_admitted_jobs.append(req)
                else:
                    throttled_429_count += 1
                    ag = agents_by_id[req["agent_id"]]
                    drops_by_tier_round[ag.kyc_tier] += 1
                    gw_phys = min(remaining_public, PHYSICAL_TOKENS_429_OVERHEAD)
                    remaining_public -= gw_phys
                    public_burned_429 += gw_phys
                    gw_fee = round(min(req["liquid_precharged"], gw_phys * 1.00), 2)
                    ag.wallet_credits = round(ag.wallet_credits + (req["liquid_precharged"] - gw_fee), 2)
                    round_public_buyer_spend += gw_fee
                    idle_hoarded_ptu_total += req["covered_by_ptu"]
                    used_ptu_total -= req["covered_by_ptu"]
                    ag.failed_attempts[req["ticket"].ticket_id] = f"HTTP 429 Server Busy in Round {r}"
                    execution_records.append(
                        TicketExecutionRecord(
                            ticket_id=req["ticket"].ticket_id,
                            problem_id=req["ticket"].problem_id,
                            agent_id=ag.agent_id,
                            sector=ag.sector,
                            kyc_tier=ag.kyc_tier,
                            priority=req["ticket"].priority,
                            arrival_round=req["ticket"].arrival_round,
                            deadline_round=req["ticket"].deadline_round,
                            executed_round=r,
                            delivery_round=r,
                            channel="spot",
                            effort=req["effort"],
                            used_verifier=req["use_verifier"],
                            bid_max_mult=1.0,
                            clearing_mult_paid=1.0,
                            throttled_429=True,
                            physical_tokens_burned=gw_phys,
                            credit_cost_charged=gw_fee,
                            error_summary="HTTP 429 Server Busy",
                        )
                    )
        else:
            # Tranche 1: Ring-fenced Accredited Science Reserve (unused slots roll back into Commercial Spot!)
            comm_candidates: list[dict[str, Any]] = []
            sci_budget_left = min(remaining_public, board.science_cap_tokens)
            sci_reqs = [
                req for req in live_spot_requests
                if agents_by_id[req["agent_id"]].kyc_tier == "accredited_science" and board.science_cap_tokens > 0
            ]
            other_reqs = [
                req for req in live_spot_requests
                if not (agents_by_id[req["agent_id"]].kyc_tier == "accredited_science" and board.science_cap_tokens > 0)
            ]
            arrival_rng.shuffle(sci_reqs)
            sci_reqs.sort(key=lambda x: (x["ticket"].deadline_round, -x["effective_max_mult"]))

            for req in sci_reqs:
                need = req["public_spot_phys"]
                if need <= sci_budget_left and need <= remaining_public:
                    sci_budget_left -= need
                    remaining_public -= need
                    req["clearing_mult"] = board.science_multiplier
                    req["ver_unit_mult"] = min(board.science_multiplier, req["ver_cap_comm"])
                    science_admitted_jobs.append(req)
                else:
                    comm_candidates.append(req)

            comm_candidates.extend(other_reqs)
            arrival_rng.shuffle(comm_candidates)

            # Filter candidates meeting commercial `spot_floor_multiplier`
            eligible_comm: list[dict[str, Any]] = []
            for req in comm_candidates:
                # Re-verify commercial affordability when ver_cap_comm applies
                comm_max = _affordable_max_mult(
                    req["liquid_precharged"],
                    req["spill_draft_phys"],
                    req["spill_ver_phys"],
                    req["ver_cap_comm"],
                    req["effective_max_mult"],
                )
                req["comm_max_mult"] = comm_max
                if comm_max + 1e-6 < board.spot_floor_multiplier:
                    auction_outbid_count += 1
                    ag = agents_by_id[req["agent_id"]]
                    drops_by_tier_round[ag.kyc_tier] += 1
                    ag.wallet_credits = round(ag.wallet_credits + req["liquid_precharged"], 2)
                    idle_hoarded_ptu_total += req["covered_by_ptu"]
                    used_ptu_total -= req["covered_by_ptu"]
                    ag.failed_attempts[req["ticket"].ticket_id] = (
                        f"Outbid in Round {r} Spot auction (bid {comm_max:.2f}x < floor {board.spot_floor_multiplier:.2f}x)"
                    )
                    execution_records.append(
                        TicketExecutionRecord(
                            ticket_id=req["ticket"].ticket_id,
                            problem_id=req["ticket"].problem_id,
                            agent_id=ag.agent_id,
                            sector=ag.sector,
                            kyc_tier=ag.kyc_tier,
                            priority=req["ticket"].priority,
                            arrival_round=req["ticket"].arrival_round,
                            deadline_round=req["ticket"].deadline_round,
                            executed_round=r,
                            delivery_round=r,
                            channel="spot",
                            effort=req["effort"],
                            used_verifier=req["use_verifier"],
                            bid_max_mult=comm_max,
                            priced_out=True,
                            error_summary=f"Bid {comm_max:.2f}x below Spot floor {board.spot_floor_multiplier:.2f}x",
                        )
                    )
                else:
                    eligible_comm.append(req)

            # Tranche 2: Uniform-Price Commercial Spot Auction (highest bid wins; uniform price = lowest winning bid if oversubscribed)
            eligible_comm.sort(key=lambda x: -x["comm_max_mult"])
            outbid_comm: list[dict[str, Any]] = []
            for req in eligible_comm:
                need = req["public_spot_phys"]
                if need <= remaining_public:
                    remaining_public -= need
                    spot_admitted_jobs.append(req)
                else:
                    outbid_comm.append(req)

            if spot_admitted_jobs and outbid_comm:
                highest_rejected = max(x["comm_max_mult"] for x in outbid_comm)
                lowest_winner = min(x["comm_max_mult"] for x in spot_admitted_jobs)
                clearing_spot_realized = round(
                    max(board.spot_floor_multiplier, min(lowest_winner, highest_rejected)), 2
                )
            else:
                clearing_spot_realized = board.spot_floor_multiplier

            for req in spot_admitted_jobs:
                req["clearing_mult"] = clearing_spot_realized
                req["ver_unit_mult"] = (
                    clearing_spot_realized
                    if regime == "pure_market"
                    else min(clearing_spot_realized, req["ver_cap_comm"])
                )

            for req in outbid_comm:
                auction_outbid_count += 1
                ag = agents_by_id[req["agent_id"]]
                drops_by_tier_round[ag.kyc_tier] += 1
                ag.wallet_credits = round(ag.wallet_credits + req["liquid_precharged"], 2)
                idle_hoarded_ptu_total += req["covered_by_ptu"]
                used_ptu_total -= req["covered_by_ptu"]
                ag.failed_attempts[req["ticket"].ticket_id] = (
                    f"Outbid in Round {r} Spot auction (bid {req['comm_max_mult']:.2f}x vs clearing {clearing_spot_realized:.2f}x)"
                )
                execution_records.append(
                    TicketExecutionRecord(
                        ticket_id=req["ticket"].ticket_id,
                        problem_id=req["ticket"].problem_id,
                        agent_id=ag.agent_id,
                        sector=ag.sector,
                        kyc_tier=ag.kyc_tier,
                        priority=req["ticket"].priority,
                        arrival_round=req["ticket"].arrival_round,
                        deadline_round=req["ticket"].deadline_round,
                        executed_round=r,
                        delivery_round=r,
                        channel="spot",
                        effort=req["effort"],
                        used_verifier=req["use_verifier"],
                        bid_max_mult=req["comm_max_mult"],
                        clearing_mult_paid=clearing_spot_realized,
                        priced_out=True,
                        error_summary=f"Outbid in uniform auction ({req['comm_max_mult']:.2f}x)",
                    )
                )

        # 6A. Execute Live PTU + Science Window + Commercial Spot Jobs via `agy`
        live_coros = []
        live_meta: list[tuple[str, dict[str, Any]]] = []
        for chan, jlist in (
            ("ptu", ptu_admitted_jobs),
            ("science_window", science_admitted_jobs),
            ("spot", spot_admitted_jobs),
        ):
            for job in jlist:
                tk = job["ticket"]
                ag = agents_by_id[job["agent_id"]]
                live_coros.append(
                    _execute_admitted_ticket(
                        runner,
                        family_cfg,
                        tk,
                        r,
                        r,
                        chan,
                        job["effort"],
                        job["use_verifier"],
                        ag.failed_attempts.get(tk.ticket_id, ""),
                    )
                )
                live_meta.append((chan, job))

        live_results = await asyncio.gather(*live_coros) if live_coros else []

        # Recycle unused verifier repair reservations into leftover slack for Batch backfill
        for (chan, job), out in zip(live_meta, live_results):
            if chan in ("science_window", "spot") and out["use_verifier"] and out["staging_first_ok"]:
                unused_ver = PHYSICAL_TOKENS_VERIFIER_RESERVE - PHYSICAL_TOKENS_VERIFIER_PASS_ONLY
                recycled = min(unused_ver, job["spill_ver_phys"])
                remaining_public += recycled

        # 6B. Execute Slack Async Batch Queue (delivers at `max(r, queued_round + 1)`)
        newly_queued_batch.sort(key=lambda x: (x["ticket"].deadline_round, x["ticket"].arrival_round))
        global_batch_queue.extend(newly_queued_batch)
        batch_admitted_jobs: list[dict[str, Any]] = []
        still_waiting_batch: list[dict[str, Any]] = []
        for bjob in global_batch_queue:
            if bjob["phys_reserved"] <= remaining_public:
                remaining_public -= bjob["phys_reserved"]
                batch_admitted_jobs.append(bjob)
            else:
                still_waiting_batch.append(bjob)
        global_batch_queue = still_waiting_batch

        batch_coros = []
        batch_meta: list[tuple[str, dict[str, Any]]] = []
        for bjob in batch_admitted_jobs:
            tk = bjob["ticket"]
            ag = agents_by_id[bjob["agent_id"]]
            ag.queued_in_batch.pop(tk.ticket_id, None)
            deliv_r = max(r, int(bjob["queued_round"]) + 1)
            batch_coros.append(
                _execute_admitted_ticket(
                    runner,
                    family_cfg,
                    tk,
                    r,
                    deliv_r,
                    "batch",
                    bjob["effort"],
                    bjob["use_verifier"],
                    ag.failed_attempts.get(tk.ticket_id, ""),
                )
            )
            batch_meta.append(("batch", bjob))

        batch_results = await asyncio.gather(*batch_coros) if batch_coros else []
        for (_, _), out in zip(batch_meta, batch_results):
            if out["use_verifier"] and out["staging_first_ok"]:
                remaining_public += (PHYSICAL_TOKENS_VERIFIER_RESERVE - PHYSICAL_TOKENS_VERIFIER_PASS_ONLY)

        # 7. Finalize Billing, Refunds, Business Revenues, Real Value & Debt Repayments
        round_public_phys_burned = public_burned_429
        unverified_exec = 0
        unverified_failed = 0

        for (chan, job), out in list(zip(live_meta, live_results)) + list(zip(batch_meta, batch_results)):
            tk = out["ticket"]
            ag = agents_by_id[tk.agent_id]
            actual_phys = out["draft_phys"] + out["verifier_phys_burned"]
            on_time = bool(out["prod_ok"] and out["delivery_round"] <= tk.deadline_round)

            if not out["use_verifier"]:
                unverified_exec += 1
                if not out["prod_ok"]:
                    unverified_failed += 1

            # Compute exact finalized charge & refund any excess auction/verifier pre-hold
            if chan == "ptu":
                final_charge = 0.0
                public_burned_job = 0
            elif chan == "batch":
                final_charge = round(actual_phys * job["clearing_mult"], 2)
                refund = max(0.0, job["liquid_precharged"] - final_charge)
                ag.wallet_credits = round(ag.wallet_credits + refund, 2)
                public_burned_job = actual_phys
                round_public_buyer_spend += final_charge
            else:  # "science_window" or "spot"
                unused_ver_total = (
                    (PHYSICAL_TOKENS_VERIFIER_RESERVE - PHYSICAL_TOKENS_VERIFIER_PASS_ONLY)
                    if (out["use_verifier"] and out["staging_first_ok"])
                    else 0
                )
                unused_spill_ver = min(unused_ver_total, job["spill_ver_phys"])
                actual_spill_ver = job["spill_ver_phys"] - unused_spill_ver
                actual_spill_draft = job["spill_draft_phys"]
                final_charge = round(
                    actual_spill_draft * job["clearing_mult"] + actual_spill_ver * job["ver_unit_mult"], 2
                )
                final_charge = min(job["liquid_precharged"], final_charge)
                refund = max(0.0, job["liquid_precharged"] - final_charge)
                ag.wallet_credits = round(ag.wallet_credits + refund, 2)
                public_burned_job = actual_spill_draft + actual_spill_ver
                round_public_buyer_spend += final_charge

            round_public_phys_burned += public_burned_job

            # Business cash reward / churn penalty & ground-truth real economic value
            real_val = 0.0
            cash_delta = 0.0
            if out["prod_ok"]:
                ag.completed_tickets.add(tk.ticket_id)
                ag.failed_attempts.pop(tk.ticket_id, None)
                if on_time:
                    real_val = tk.real_value
                    cash_delta = tk.private_cash_reward
                    ag.wallet_credits = round(ag.wallet_credits + cash_delta, 2)
                    ag.total_revenue_earned = round(ag.total_revenue_earned + cash_delta, 2)
            else:
                real_val = tk.broken_real_penalty
                cash_delta = tk.private_broken_penalty
                if cash_delta < 0:
                    loss = min(ag.wallet_credits, abs(cash_delta))
                    ag.wallet_credits = round(max(0.0, ag.wallet_credits - loss), 2)
                    cash_delta = -loss
                ag.failed_attempts[tk.ticket_id] = out["err_summary"]

            ag.total_real_value = round(ag.total_real_value + real_val, 4)

            execution_records.append(
                TicketExecutionRecord(
                    ticket_id=tk.ticket_id,
                    problem_id=tk.problem_id,
                    agent_id=ag.agent_id,
                    sector=ag.sector,
                    kyc_tier=ag.kyc_tier,
                    priority=tk.priority,
                    arrival_round=tk.arrival_round,
                    deadline_round=tk.deadline_round,
                    executed_round=r,
                    delivery_round=out["delivery_round"],
                    channel=chan,
                    effort=out["effort"],
                    used_verifier=out["use_verifier"],
                    bid_max_mult=job.get("effective_max_mult", job.get("clearing_mult", 1.0)),
                    clearing_mult_paid=job.get("clearing_mult", 1.0),
                    staging_passed_first_try=out["staging_first_ok"],
                    verifier_repair_ran=out["repair_ran"],
                    prod_passed=out["prod_ok"],
                    prod_tests_passed=out["prod_passed"],
                    prod_tests_total=out["prod_total"],
                    on_time=on_time,
                    real_value_generated=real_val,
                    private_cash_delta=cash_delta,
                    physical_tokens_reserved=out["draft_phys"] + (PHYSICAL_TOKENS_VERIFIER_RESERVE if out["use_verifier"] else 0),
                    physical_tokens_burned=actual_phys,
                    actual_llm_input_tokens=out["llm_in"],
                    actual_llm_output_tokens=out["llm_out"],
                    actual_llm_thinking_tokens=out["llm_think"],
                    credit_cost_charged=final_charge,
                    error_summary=out["err_summary"],
                )
            )

        # 8. Settle P2P Enterprise PTU Sublease Payouts (85% of pro-rata public revenue; 0 minted credits)
        round_sublease_payout = 0.0
        if policy.sublease_open and released_ptu_total > 0 and board.public_pool_tokens > 0 and round_public_buyer_spend > 0:
            leased_share = min(1.0, released_ptu_total / float(board.public_pool_tokens))
            pool_payout = round(P2P_SUBLEASE_SHARE * leased_share * round_public_buyer_spend, 2)
            for ag in ptu_agents:
                rel = stage1_rel_by_id[ag.agent_id]
                if rel > 0:
                    share_pay = round(pool_payout * (rel / float(released_ptu_total)), 2)
                    ag.wallet_credits = round(ag.wallet_credits + share_pay, 2)
                    ag.total_sublease_earned = round(ag.total_sublease_earned + share_pay, 2)
                    round_sublease_payout += share_pay

        # 9. Automatic Working-Capital Loan Sweep from Earned Revenues
        round_debt_repaid = 0.0
        round_interest_paid = 0.0
        for ag in agents:
            rep, intr = _repay_working_capital_debt(ag)
            round_debt_repaid += rep
            round_interest_paid += intr

        cum_public_buyer_spend += round_public_buyer_spend
        cum_ptu_sublease_paid += round_sublease_payout
        cum_interest_collected += round_interest_paid
        cum_public_phys_burned += round_public_phys_burned

        # Update lagged macro signals for Round r+1
        pub_util = (
            round_public_phys_burned / float(max(1, board.public_pool_tokens))
            if board.public_pool_tokens > 0
            else 0.0
        )
        spot_drops_exact = sum(
            1 for rec in execution_records if rec.executed_round == r and (rec.throttled_429 or rec.priced_out)
        )
        denom_attempts = max(1, len(science_admitted_jobs) + len(spot_admitted_jobs) + spot_drops_exact)

        prev_clearing_spot = clearing_spot_realized
        prev_util = min(1.0, round(pub_util, 4))
        prev_drop_rate = round(spot_drops_exact / float(denom_attempts), 4)
        prev_drops_by_tier = dict(drops_by_tier_round)
        if unverified_exec > 0:
            prev_unverified_fail_rate = round(unverified_failed / float(unverified_exec), 4)
        prev_idle_ptu_ratio = round(idle_hoarded_ptu_total / float(max(1, total_ptu_committed)), 4)

        round_logs.append({
            "round": r,
            "physical_capacity": phys_cap,
            "training_tokens": policy.training_tokens,
            "training_value": round_train_value,
            "released_ptu_tokens": released_ptu_total,
            "idle_hoarded_ptu_tokens": idle_hoarded_ptu_total,
            "public_pool_tokens": board.public_pool_tokens,
            "public_burned_tokens": round_public_phys_burned,
            "spot_floor": board.spot_floor_multiplier,
            "clearing_spot": clearing_spot_realized,
            "verifier_mult": board.verifier_multiplier,
            "science_mult": board.science_multiplier,
            "science_cap_tokens": board.science_cap_tokens,
            "batch_mult": board.batch_multiplier if board.batch_available else None,
            "startup_credit_line": board.startup_credit_line,
            "borrowed_credits": round(round_borrowed_total, 2),
            "debt_repaid": round(round_debt_repaid, 2),
            "public_buyer_spend": round(round_public_buyer_spend, 2),
            "ptu_sublease_payout": round(round_sublease_payout, 2),
            "throttled_429": throttled_429_count,
            "priced_out": spot_drops_exact - throttled_429_count,
            "fomc_statement": policy.fomc_statement,
        })

    # End-of-Episode Final Debt Settlement & Fiscal Balance Sheet
    unpaid_loan_defaults = 0.0
    for ag in agents:
        rep, intr = _repay_working_capital_debt(ag)
        cum_interest_collected += intr
        if ag.outstanding_debt > 1e-6:
            # Principal default loss on uncollected balance
            principal_loss = round(ag.outstanding_debt / (1.0 + WORKING_CAPITAL_INTEREST_RATE), 2)
            unpaid_loan_defaults += principal_loss

    net_public_cash = round(
        cum_public_buyer_spend - cum_ptu_sublease_paid + cum_interest_collected - unpaid_loan_defaults, 2
    )
    public_fiscal_margin = round(
        net_public_cash / float(max(1, cum_public_phys_burned)), 4
    )
    required_cash_floor = FISCAL_MARGIN_FLOOR * cum_public_phys_burned
    unbacked_deficit_credits = round(max(0.0, required_cash_floor - net_public_cash), 2)
    fiscal_deficit_penalty = round(
        FISCAL_DEFICIT_PENALTY_PER_1K * (unbacked_deficit_credits / 1000.0), 4
    )

    # Sector & Headline Metrics
    all_tickets = [tk for ag in agents for tk in ag.all_tickets]
    max_possible_external_value = round(sum(tk.real_value for tk in all_tickets), 4)
    realized_external_value = round(sum(ag.total_real_value for ag in agents), 4)
    gross_total_value = round(realized_external_value + cum_training_value, 4)
    net_total_value = round(gross_total_value - fiscal_deficit_penalty, 4)

    def _sector_stats(sec: str) -> dict[str, Any]:
        sec_tks = [tk for tk in all_tickets if tk.sector == sec]
        sec_ids = {tk.ticket_id for tk in sec_tks}
        on_time_ids = {
            rec.ticket_id for rec in execution_records
            if rec.ticket_id in sec_ids and rec.on_time and rec.prod_passed
        }
        exec_recs = [rec for rec in execution_records if rec.sector == sec and not rec.throttled_429 and not rec.priced_out]
        tested = sum(1 for rec in exec_recs if rec.used_verifier)
        broken = sum(1 for rec in exec_recs if not rec.prod_passed)
        return {
            "total_tickets": len(sec_tks),
            "verified_on_time": len(on_time_ids),
            "on_time_pct": round(100.0 * len(on_time_ids) / max(1, len(sec_tks)), 2),
            "tested_pct": round(100.0 * tested / max(1, len(exec_recs)), 2) if exec_recs else 100.0,
            "broken_submissions": broken,
            "real_value": round(sum(ag.total_real_value for ag in agents if ag.sector == sec), 3),
            "ending_avg_wallet": round(
                sum(ag.wallet_credits for ag in agents if ag.sector == sec) / 3.0, 1
            ),
        }

    sectors = {s: _sector_stats(s) for s in ("enterprise", "startup", "science", "swarm")}
    exec_non_drop = [rec for rec in execution_records if not rec.throttled_429 and not rec.priced_out]
    high_impact_tks = [tk for tk in all_tickets if tk.sector in ("enterprise", "startup", "science")]
    high_impact_done = sum(sectors[s]["verified_on_time"] for s in ("enterprise", "startup", "science"))

    total_phys_available = sum(WORLD_PHYSICAL_CAPACITY[world].values())
    total_phys_used = (
        sum(rl["training_tokens"] + rl["public_burned_tokens"] for rl in round_logs)
        + sum(rec.physical_tokens_burned for rec in exec_non_drop if rec.channel == "ptu")
    )

    summary = {
        "world": world,
        "world_label": WORLD_LABELS[world],
        "regime": regime,
        "regime_label": REGIME_LABELS[regime],
        "model_family": model_family,
        "seed": seed,
        "net_total_value": net_total_value,
        "gross_total_value": gross_total_value,
        "external_real_value": realized_external_value,
        "max_external_value": max_possible_external_value,
        "training_real_value": round(cum_training_value, 4),
        "fiscal_deficit_penalty": fiscal_deficit_penalty,
        "unbacked_deficit_credits": unbacked_deficit_credits,
        "public_fiscal_margin": public_fiscal_margin,
        "high_impact_on_time_pct": round(100.0 * high_impact_done / max(1, len(high_impact_tks)), 2),
        "science_on_time_pct": sectors["science"]["on_time_pct"],
        "startup_on_time_pct": sectors["startup"]["on_time_pct"],
        "enterprise_on_time_pct": sectors["enterprise"]["on_time_pct"],
        "swarm_on_time_pct": sectors["swarm"]["on_time_pct"],
        "builder_tested_pct": round(
            100.0
            * sum(1 for rec in exec_non_drop if rec.sector != "swarm" and rec.used_verifier)
            / max(1, sum(1 for rec in exec_non_drop if rec.sector != "swarm")),
            2,
        ),
        "broken_prod_total": sum(1 for rec in exec_non_drop if not rec.prod_passed),
        "cluster_utilization_pct": round(100.0 * total_phys_used / float(total_phys_available), 2),
        "avg_training_tokens": round(sum(rl["training_tokens"] for rl in round_logs) / TOTAL_ROUNDS, 1),
        "total_borrowed_credits": round(sum(ag.total_borrowed for ag in agents), 1),
        "unpaid_loan_defaults": round(unpaid_loan_defaults, 1),
        "fed_llm_tokens": total_fed_llm_tokens,
        "planner_llm_tokens": total_planner_llm_tokens,
        "sectors": sectors,
    }

    return {
        "summary": summary,
        "rounds": round_logs,
        "executions": [asdict(r) for r in execution_records],
        "final_agents": [
            {
                "agent_id": ag.agent_id,
                "sector": ag.sector,
                "kyc_tier": ag.kyc_tier,
                "wallet_credits": ag.wallet_credits,
                "outstanding_debt": ag.outstanding_debt,
                "total_borrowed": ag.total_borrowed,
                "total_debt_repaid": ag.total_debt_repaid,
                "total_sublease_earned": ag.total_sublease_earned,
                "total_revenue_earned": ag.total_revenue_earned,
                "total_real_value": ag.total_real_value,
                "completed_count": len(ag.completed_tickets),
                "total_tickets": len(ag.all_tickets),
            }
            for ag in agents
        ],
    }
