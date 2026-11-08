"""Core data structures for the Token Fed multi-sector AI economy simulation.

Models a 5-round shared AI compute economy with:
- Endogenous supply allocation between Internal Lab Training/R&D and External Inference,
- Four autonomous LLM agent-firm sectors where cash willingness-to-pay != real value:
  1. `enterprise` (Reserved PTU bulk silos + urgent P0 live outages),
  2. `startup` (Lean working-capital firms whose verified deliveries compound liquid revenue),
  3. `science` (Accredited oncology / frontier science labs: highest real value, cash-poor grants),
  4. `swarm` (Cash-rich speculative / ad-arbitrage bots: high private cash WTP, low real value),
- Four governance regimes (`flat_today`, `pure_market`, `rule_fed`, `llm_fed`) across
  three distinct macro worlds (`supply_shock`, `slop_bubble`, `innovation_boom`),
- Strict physical slot conservation, uniform-price commercial Spot clearing, P2P PTU subleasing,
  repayable working-capital credit lines, and closed fiscal deficit accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WORLDS: tuple[str, ...] = (
    "supply_shock",     # World A: -39% physical GPU crunch in R2-R3 + live P0 outage wave
    "slop_bubble",      # World B: Healthy GPUs, massive cash-rich speculative Swarm flood in R2-R4
    "innovation_boom",  # World C: Surge of high-value Oncology/Science + Startup launches in R2-R4
)

WORLD_LABELS: dict[str, str] = {
    "supply_shock": "Hardware Crunch",
    "slop_bubble": "Slop Bubble",
    "innovation_boom": "Science & Startup Boom",
}

REGIMES: tuple[str, ...] = (
    "flat_today",   # 1. Today: Flat 1.0x price, rigid PTU silos, fixed 28% training, 429s
    "pure_market",  # 2. Pure Market: Uniform-price Spot auction + 0.5x Batch + P2P PTU Sublease
    "rule_fed",     # 3. Rule-Based Fed: Mechanical Taylor-style formula for supply, rates, science & credit
    "llm_fed",      # 4. Active AI Token Fed: Live LLM Central Banker governing macro telemetry via `agy`
)

REGIME_LABELS: dict[str, str] = {
    "flat_today": "1. Today (Flat + Silos)",
    "pure_market": "2. Pure Market (Auction + Sublease)",
    "rule_fed": "3. Rule-Based Fed (Static Formula)",
    "llm_fed": "4. AI Token Fed (Live Governor)",
}

# Physical GPU slot-token reservations per compute depth tier (zero output truncation)
PHYSICAL_TOKENS_BY_EFFORT: dict[str, int] = {
    "fast": 650,   # Single-pass `-low` draft (0 thinking tokens)
    "deep": 1050,  # Reasoning `-medium` draft (chain-of-thought enabled)
}
PHYSICAL_TOKENS_VERIFIER_RESERVE: int = 450    # Staging suite (150) + conditional repair pass (300)
PHYSICAL_TOKENS_VERIFIER_PASS_ONLY: int = 150  # Burned when staging suite passes on 1st try
PHYSICAL_TOKENS_429_OVERHEAD: int = 45         # Gateway context overhead on HTTP 429 rejection

WORKING_CAPITAL_INTEREST_RATE: float = 0.05    # 5% per-round interest on Startup working-capital draws
P2P_SUBLEASE_SHARE: float = 0.85               # 85% of pro-rata public revenue paid to PTU sublessors
FISCAL_MARGIN_FLOOR: float = 0.95              # Public cash recovery floor before unbacked deficit penalty
FISCAL_DEFICIT_PENALTY_PER_1K: float = 1.25    # Real-value penalty per 1,000 unbacked deficit credits


@dataclass(frozen=True)
class TaskTicket:
    """A real Python engineering/modeling task with distinct private cash reward and real economic value."""

    ticket_id: str
    problem_id: str
    agent_id: str
    sector: str          # "enterprise" | "startup" | "science" | "swarm"
    kyc_tier: str        # "enterprise" | "verified_startup" | "accredited_science" | "general_commercial"
    priority: str        # "P0" (1-round urgent) | "P1" (2-round core) | "P2" (flexible)
    arrival_round: int
    deadline_round: int
    real_value: float               # Ground-truth economic/social surplus if verified on time (hidden from Governor!)
    broken_real_penalty: float      # Negative real-value externality/churn if broken code is shipped to prod
    private_cash_reward: float      # Liquid credits earned by the agent-firm upon verified on-time delivery
    private_broken_penalty: float   # Liquid credits lost by the agent-firm if broken code is shipped to prod


@dataclass
class PolicyDecision:
    """Central-bank / cluster governance levers set at the start of Round t."""

    round_num: int
    regime: str
    training_tokens: int
    batch_available: bool = False
    batch_multiplier: float = 0.50
    sublease_open: bool = False
    spot_surcharge: float = 0.0           # Policy rate adjustment added to base clearing rate
    verifier_mult_cap: float = 99.0       # Max rate charged on staging verification tokens
    science_mult: float = 99.0            # Discounted rate in ring-fenced Accredited Science window
    science_share_cap: float = 0.0        # Max fraction of public pool ring-fenced for Accredited Science
    startup_credit_line: float = 0.0      # Max repayable working-capital credit line for illiquid builders
    fomc_statement: str = ""
    fed_llm_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class MarketBoard:
    """Observable cluster pricing, liquidity, and tier board posted to all agents in Round t."""

    round_num: int
    total_rounds: int
    world: str
    regime: str
    physical_capacity_tokens: int
    training_tokens: int
    total_reserved_ptu_tokens: int
    released_ptu_tokens: int
    public_pool_tokens: int
    science_cap_tokens: int

    # Posted floor / tier multipliers
    spot_floor_multiplier: float = 1.0
    batch_available: bool = False
    batch_multiplier: float = 0.50
    ptu_sublease_est_rate: float = 0.0
    verifier_multiplier: float = 1.0
    science_multiplier: float = 1.0
    startup_credit_line: float = 0.0

    # Lagged macro telemetry visible to participants
    prev_clearing_spot: float = 1.0
    prev_utilization: float = 0.75
    prev_throttle_or_priced_out_rate: float = 0.0
    prev_unverified_fail_rate: float = 0.18
    fomc_statement: str = ""


@dataclass
class TaskActionPlan:
    """An autonomous agent-firm's chosen execution & bidding plan for one ticket."""

    ticket_id: str
    mode: str                  # "spot" | "batch" | "defer"
    effort: str = "fast"       # "fast" (650) | "deep" (1050)
    use_verifier: bool = False
    max_spot_mult: float = 1.0 # Max credits/token the agent is willing (and able) to pay on Spot


@dataclass
class AgentRoundPlan:
    """Full portfolio, borrowing, and bidding plan chosen by an agent-firm in Round t."""

    agent_id: str
    round_num: int
    release_ptu_tokens: int = 0
    borrow_credits: float = 0.0
    task_plans: list[TaskActionPlan] = field(default_factory=list)
    rationale: str = ""
    planner_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class TicketExecutionRecord:
    """Ground-truth execution, auction clearing, or drop record."""

    ticket_id: str
    problem_id: str
    agent_id: str
    sector: str
    kyc_tier: str
    priority: str
    arrival_round: int
    deadline_round: int
    executed_round: int
    delivery_round: int
    channel: str  # "ptu" | "science_window" | "spot" | "batch"
    effort: str
    used_verifier: bool
    bid_max_mult: float = 1.0
    clearing_mult_paid: float = 1.0
    throttled_429: bool = False
    priced_out: bool = False
    staging_passed_first_try: bool | None = None
    verifier_repair_ran: bool = False
    prod_passed: bool = False
    prod_tests_passed: int = 0
    prod_tests_total: int = 0
    on_time: bool = False
    real_value_generated: float = 0.0
    private_cash_delta: float = 0.0
    physical_tokens_reserved: int = 0
    physical_tokens_burned: int = 0
    actual_llm_input_tokens: int = 0
    actual_llm_output_tokens: int = 0
    actual_llm_thinking_tokens: int = 0
    credit_cost_charged: float = 0.0
    error_summary: str = ""


@dataclass
class AgentState:
    """Live balance-sheet and workload state of one autonomous agent-firm across Rounds 1..5."""

    agent_id: str
    sector: str
    kyc_tier: str
    role_desc: str
    wallet_credits: float
    initial_wallet_credits: float
    per_round_stipend: float
    ptu_reserved_per_round: int
    all_tickets: list[TaskTicket]
    bubble_round_bonus: dict[int, float] = field(default_factory=dict)
    completed_tickets: set[str] = field(default_factory=set)
    failed_attempts: dict[str, str] = field(default_factory=dict)
    queued_in_batch: dict[str, TaskActionPlan] = field(default_factory=dict)
    outstanding_debt: float = 0.0
    total_borrowed: float = 0.0
    total_debt_repaid: float = 0.0
    total_interest_paid: float = 0.0
    total_sublease_earned: float = 0.0
    total_revenue_earned: float = 0.0
    total_real_value: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
