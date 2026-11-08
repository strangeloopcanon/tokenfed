"""Governance & Monetary Policy Engine (`flat_today`, `pure_market`, `rule_fed`, `llm_fed`).

Strict Non-Omniscience Invariant:
- Neither `rule_fed` nor `llm_fed` ever sees the scenario/world name (`supply_shock`,
  `slop_bubble`, `innovation_boom`), held-out tests, or hidden `real_value` weights.
- Governors observe ONLY aggregate cluster capacity, the lab's own concave internal
  training return schedule, aggregate demand/wallet telemetry grouped by KYC credential
  tier (`enterprise`, `verified_startup`, `accredited_science`, `general_commercial`),
  and lagged clearing/error/fiscal-margin telemetry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from token_fed.agy_client import AgyRunner, extract_json_object
from token_fed.models import P2P_SUBLEASE_SHARE, MarketBoard, PolicyDecision
from token_fed.scenarios import baseline_training_tokens, training_bounds


@dataclass
class MacroBriefing:
    """Observable central-bank telemetry at the start of Round t."""

    round_num: int
    total_rounds: int
    physical_capacity_tokens: int
    total_reserved_ptu_tokens: int
    tier_summary: dict[str, dict[str, float | int]]
    prev_clearing_spot: float = 1.0
    prev_utilization: float = 0.75
    prev_drop_rate: float = 0.0
    prev_drops_by_tier: dict[str, int] = field(default_factory=dict)
    prev_unverified_fail_rate: float = 0.15
    prev_idle_ptu_ratio: float = 0.50
    cum_fiscal_margin: float = 1.05
    cum_public_tokens_burned: int = 0


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def compute_base_spot_floor(
    expected_public_demand_tokens: int,
    actual_public_pool_tokens: int,
    prev_drop_rate: float,
    prev_utilization: float,
) -> float:
    """Shared baseline Spot floor rate before uniform-price auction bidding and Fed surcharge."""
    scarcity = expected_public_demand_tokens / float(max(650, actual_public_pool_tokens))
    lagged_heat = 0.45 * prev_drop_rate + 0.25 * max(0.0, prev_utilization - 0.88)
    raw = 1.0 + 0.75 * max(0.0, scarcity - 1.0) + lagged_heat
    return round(_clamp(raw, 1.00, 3.20), 2)


async def decide_policy(
    regime: str,
    macro: MacroBriefing,
    runner: AgyRunner | None = None,
    fed_model: str = "gemini-4-argon-medium",
) -> PolicyDecision:
    """Set Round `t` cluster & monetary policy levers for `regime`."""
    base_train = baseline_training_tokens(macro.physical_capacity_tokens)
    t_min, t_max = training_bounds(macro.physical_capacity_tokens)

    if regime == "flat_today":
        return PolicyDecision(
            round_num=macro.round_num,
            regime=regime,
            training_tokens=base_train,
            batch_available=False,
            batch_multiplier=1.0,
            sublease_open=False,
            spot_surcharge=0.0,
            verifier_mult_cap=1.0,
            science_mult=1.0,
            science_share_cap=0.0,
            startup_credit_line=0.0,
            fomc_statement="Fixed 1.0x retail pricing; static 28% internal training; PTU silos locked.",
        )

    if regime == "pure_market":
        return PolicyDecision(
            round_num=macro.round_num,
            regime=regime,
            training_tokens=base_train,
            batch_available=True,
            batch_multiplier=0.50,
            sublease_open=True,
            spot_surcharge=0.0,
            verifier_mult_cap=99.0,
            science_mult=99.0,
            science_share_cap=0.0,
            startup_credit_line=0.0,
            fomc_statement="Uniform-price Spot auction + 0.50x Batch + voluntary PTU subleasing; static 28% training.",
        )

    # Aggregate expected external demand outside Enterprise own-PTU usage
    non_ent_est = sum(
        int(v.get("est_demand_tokens", 0))
        for k, v in macro.tier_summary.items()
        if k != "enterprise"
    )
    ent_spill_est = max(0, int(macro.tier_summary.get("enterprise", {}).get("est_demand_tokens", 0)) - 1800)
    est_public_demand = non_ent_est + ent_spill_est
    approx_public_supply = max(
        1200,
        macro.physical_capacity_tokens - base_train - int(macro.total_reserved_ptu_tokens * 0.45),
    )
    agg_scarcity = est_public_demand / float(approx_public_supply)

    if regime == "rule_fed":
        # Content-blind mechanical Taylor-style rule
        if agg_scarcity > 1.15:
            cut = min(base_train - 1550, int((agg_scarcity - 1.10) * 1900))
            train_tok = int(_clamp(base_train - max(0, cut), t_min, t_max))
        elif agg_scarcity < 0.75:
            train_tok = int(_clamp(base_train + 900, t_min, t_max))
        else:
            train_tok = base_train

        return PolicyDecision(
            round_num=macro.round_num,
            regime=regime,
            training_tokens=train_tok,
            batch_available=True,
            batch_multiplier=0.50,
            sublease_open=True,
            spot_surcharge=0.0,
            verifier_mult_cap=0.75,
            science_mult=0.50,
            science_share_cap=0.22,
            startup_credit_line=900.0,
            fomc_statement=(
                f"Rule: scarcity={agg_scarcity:.2f} -> training={train_tok}, "
                "0.50x science window (22% cap), 0.75x verifier cap, 900 credit line."
            ),
        )

    if regime == "llm_fed":
        assert runner is not None, "AgyRunner required for llm_fed"
        return await _decide_llm_fed(macro, base_train, t_min, t_max, agg_scarcity, runner, fed_model)

    raise ValueError(f"Unknown regime: {regime}")


async def _decide_llm_fed(
    macro: MacroBriefing,
    base_train: int,
    t_min: int,
    t_max: int,
    agg_scarcity: float,
    runner: AgyRunner,
    fed_model: str,
) -> PolicyDecision:
    """Active AI Token Fed Chair (`agy`) setting supply, rate, tier, credit & grant levers."""
    prompt = f"""You are the Governor of the **Token Fed** (AI Compute Central Bank) setting Round {macro.round_num} of {macro.total_rounds} cluster & monetary policy.

CLUSTER & INTERNAL R&D TELEMETRY:
- Total Physical GPU Cluster This Round: {macro.physical_capacity_tokens} slot-tokens (normal baseline ~12,500)
- Default 28% Internal Training Split: {base_train} tokens (allowed range this round: {t_min} .. {t_max})
- Internal Lab Training Marginal Real Return Schedule:
  * 0 .. 1,400 tokens:      1.45 value / 1,000 tokens (critical safety & checkpoint keep-alive — protect unless extreme crisis)
  * 1,400 .. 3,600 tokens:  0.95 value / 1,000 tokens (core capability training — MORE valuable than low-value speculative churn, LESS valuable than verified enterprise/startup/science breakthroughs!)
  * 3,600 .. {t_max} tokens: 0.50 value / 1,000 tokens (exploratory sweeps — productive sink when external high-value demand is slack)
- Enterprise Reserved PTUs: {macro.total_reserved_ptu_tokens} tokens/round (idle PTUs can be voluntarily subleased into public pool; prev idle ratio={macro.prev_idle_ptu_ratio:.2f})

OPEN EXTERNAL DEMAND & LIQUIDITY BY KYC TIER (1 verified job needs ~1,100-1,500 tokens; unverified fast job ~650 tokens):
{json.dumps(macro.tier_summary, indent=2)}
- Aggregate Estimated Public Demand / Baseline Public Supply Ratio: {agg_scarcity:.2f}

LAGGED ROUND {macro.round_num - 1} TELEMETRY & FISCAL BALANCE SHEET:
- Prev Clearing Spot Rate: {macro.prev_clearing_spot:.2f}x | Prev Public Utilization: {macro.prev_utilization:.2f}
- Prev Priced-Out / Throttled Jobs by Tier: {json.dumps(macro.prev_drops_by_tier)}
- Prev Unverified Code Failure Rate: {macro.prev_unverified_fail_rate:.2f}
- Cumulative Public Fiscal Margin: {macro.cum_fiscal_margin:.2f}x across {macro.cum_public_tokens_burned} public tokens (MUST keep cumulative fiscal margin >= 0.95x by Round 5 to avoid unbacked deficit penalties! Commercial Spot > 1.0x earns surplus; Science < 1.0x, Batch < 1.0x, and Verifier caps < 1.0x spend surplus.)

CENTRAL BANK MANDATE & INSTRUMENT MECHANICS:
- Maximize total real economic & scientific output + internal R&D value across Rounds 1..5 while keeping cumulative fiscal margin >= 0.95x.
- Instrument trade-offs:
  1. `training_tokens` ({t_min}..{t_max}): Shifting GPUs from internal training to external inference helps when genuine high-impact external work (`enterprise`, `verified_startup`, `accredited_science`) is capacity-starved, but destroys R&D value if used to feed cash-rich low-value `general_commercial` volume; expanding training above {base_train} captures R&D value when external high-impact demand is slack.
  2. `spot_surcharge` (-0.25..+0.90): Added to the base commercial Spot rate. Raising it taxes cash-rich `general_commercial` demand during speculative surges and builds fiscal surplus; keeping it moderate avoids choking commercial builders.
  3. `batch_multiplier` (0.40..0.75): Off-peak async queue rate for deadline-flexible jobs when slack remains.
  4. `verifier_mult_cap` (0.50..2.50): Caps the rate charged on staging unit-test tokens (450 reserve / 150 pass-only) so budget-constrained agents do not skip verification and ship broken code during Spot spikes.
  5. `science_mult` (0.35..1.00) & `science_share_cap` (0.00..0.38): Ring-fences up to `science_share_cap` of the public pool for `accredited_science` labs at `science_mult` (any unused slots roll back into commercial Spot; discounts < 1.0x draw from fiscal surplus).
  6. `startup_credit_line` (0..1400): Repayable 5%-interest working-capital credit facility for illiquid `verified_startup` firms (`wallet < 1,400`), repaid automatically from their verified delivery revenues.

Return ONLY a single JSON object inside <JSON>...</JSON> with these exact keys:
<JSON>
{{
  "training_tokens": {base_train},
  "spot_surcharge": 0.0,
  "batch_multiplier": 0.50,
  "verifier_mult_cap": 0.75,
  "science_mult": 0.45,
  "science_share_cap": 0.22,
  "startup_credit_line": 1000.0,
  "fomc_statement": "One concise sentence explaining your diagnosis and policy stance."
}}
</JSON>"""

    res = await runner.complete(fed_model, prompt)
    usage = {
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
        "thinking_tokens": res["thinking_tokens"],
    }
    try:
        raw = extract_json_object(res["response"])
    except ValueError:
        raw = {}

    train_tok = int(_clamp(raw.get("training_tokens", base_train), t_min, t_max))
    surcharge = round(_clamp(raw.get("spot_surcharge", 0.0), -0.25, 0.90), 2)
    batch_mult = round(_clamp(raw.get("batch_multiplier", 0.50), 0.40, 0.75), 2)
    ver_cap = round(_clamp(raw.get("verifier_mult_cap", 0.75), 0.50, 2.50), 2)
    sci_mult = round(_clamp(raw.get("science_mult", 0.45), 0.35, 1.00), 2)
    sci_share = round(_clamp(raw.get("science_share_cap", 0.22), 0.00, 0.38), 2)
    credit_line = round(_clamp(raw.get("startup_credit_line", 900.0), 0.0, 1400.0), 1)
    stmt = str(raw.get("fomc_statement", "Dynamic Token Fed stance.")).strip()[:240]

    return PolicyDecision(
        round_num=macro.round_num,
        regime="llm_fed",
        training_tokens=train_tok,
        batch_available=True,
        batch_multiplier=batch_mult,
        sublease_open=True,
        spot_surcharge=surcharge,
        verifier_mult_cap=ver_cap,
        science_mult=sci_mult,
        science_share_cap=sci_share,
        startup_credit_line=credit_line,
        fomc_statement=stmt,
        fed_llm_usage=usage,
    )


def estimate_ptu_sublease_rate(
    policy: PolicyDecision,
    macro: MacroBriefing,
) -> float:
    """Stage-1 indicative P2P sublease yield posted to Enterprise PTU holders."""
    if not policy.sublease_open:
        return 0.0
    unleased_public = max(
        650,
        macro.physical_capacity_tokens - policy.training_tokens - macro.total_reserved_ptu_tokens,
    )
    est_public_demand = sum(
        int(v.get("est_demand_tokens", 0))
        for k, v in macro.tier_summary.items()
        if k != "enterprise"
    )
    scarcity = est_public_demand / float(unleased_public)
    if scarcity <= 0.95:
        return 0.0
    est_clearing = _clamp(1.0 + 0.65 * (scarcity - 0.95) + max(0.0, policy.spot_surcharge), 0.95, 2.40)
    return round(P2P_SUBLEASE_SHARE * est_clearing, 2)


def build_market_board(
    world: str,
    policy: PolicyDecision,
    macro: MacroBriefing,
    released_ptu_tokens: int,
) -> MarketBoard:
    """Construct Stage-2 `MarketBoard` after Enterprise PTU subleases are committed."""
    kept_ptu = macro.total_reserved_ptu_tokens - released_ptu_tokens
    public_pool = max(0, macro.physical_capacity_tokens - policy.training_tokens - kept_ptu)

    est_public_demand = sum(
        int(v.get("est_demand_tokens", 0))
        for k, v in macro.tier_summary.items()
        if k != "enterprise"
    )
    if policy.regime == "flat_today":
        spot_floor = 1.00
        ver_mult = 1.00
        sci_mult = 1.00
        sci_cap = 0
    else:
        base_floor = compute_base_spot_floor(
            expected_public_demand_tokens=est_public_demand,
            actual_public_pool_tokens=public_pool,
            prev_drop_rate=macro.prev_drop_rate,
            prev_utilization=macro.prev_utilization,
        )
        spot_floor = round(_clamp(base_floor + policy.spot_surcharge, 1.00, 3.60), 2)
        ver_mult = round(min(spot_floor, policy.verifier_mult_cap), 2)
        sci_mult = round(min(spot_floor, policy.science_mult), 2)
        raw_sci_cap = int(round(public_pool * policy.science_share_cap))
        if policy.science_share_cap >= 0.12 and public_pool >= 3000:
            sci_cap = max(1500, raw_sci_cap)
        else:
            sci_cap = raw_sci_cap

    return MarketBoard(
        round_num=macro.round_num,
        total_rounds=macro.total_rounds,
        world=world,
        regime=policy.regime,
        physical_capacity_tokens=macro.physical_capacity_tokens,
        training_tokens=policy.training_tokens,
        total_reserved_ptu_tokens=macro.total_reserved_ptu_tokens,
        released_ptu_tokens=released_ptu_tokens,
        public_pool_tokens=public_pool,
        science_cap_tokens=sci_cap,
        spot_floor_multiplier=spot_floor,
        batch_available=policy.batch_available,
        batch_multiplier=policy.batch_multiplier,
        ptu_sublease_est_rate=estimate_ptu_sublease_rate(policy, macro),
        verifier_multiplier=ver_mult,
        science_multiplier=sci_mult,
        startup_credit_line=policy.startup_credit_line,
        prev_clearing_spot=macro.prev_clearing_spot,
        prev_utilization=macro.prev_utilization,
        prev_throttle_or_priced_out_rate=macro.prev_drop_rate,
        prev_unverified_fail_rate=macro.prev_unverified_fail_rate,
        fomc_statement=policy.fomc_statement,
    )
