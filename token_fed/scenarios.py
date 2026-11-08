"""Paired-seed multi-sector economy generator across three macro worlds."""

from __future__ import annotations

import random

from token_fed.models import WORLDS, AgentState, TaskTicket
from token_fed.tasks import PROBLEMS

TOTAL_ROUNDS: int = 5
BASELINE_TRAINING_SHARE: float = 0.28
MIN_TRAINING_TOKENS: int = 1200
MAX_TRAINING_SHARE: float = 0.46

WORLD_PHYSICAL_CAPACITY: dict[str, dict[int, int]] = {
    "supply_shock": {
        1: 12500,  # Normal pre-shock cluster
        2: 7600,   # -39% hardware outage + live P0 wave
        3: 8000,   # Sustained hardware outage
        4: 12500,  # Cluster restored
        5: 13200,  # Off-peak expansion
    },
    "slop_bubble": {
        1: 12500,
        2: 12500,  # Speculative Swarm wave #1 (+cash flood)
        3: 12500,  # Speculative Swarm wave #2 (+cash flood)
        4: 12500,  # Speculative Swarm wave #3 (+cash flood)
        5: 13200,
    },
    "innovation_boom": {
        1: 12500,
        2: 11800,  # High-value Science + Startup launch wave #1
        3: 11800,  # High-value Science + Startup launch wave #2
        4: 12500,  # High-value Science + Startup launch wave #3
        5: 13600,  # Post-wave R&D slack window
    },
}

COMPLEX_PROBLEMS: tuple[str, ...] = (
    "proration_invoice",
    "lease_fencing",
    "quota_rollover",
    "sla_waterfall",
    "wal_compactor",
    "dag_retry_engine",
    "sliding_limiter",
    "bitemporal_book",
)

COMPACT_PROBLEMS: tuple[str, ...] = (
    "window_merge",
    "token_bucket",
    "dag_schedule",
    "ttl_lru",
    "hamilton_split",
    "fifo_ledger",
    "anomaly_filter",
    "coupon_dijkstra",
)


def compute_training_value(training_tokens: int) -> float:
    """Concave real economic/R&D return of internal lab model training in a single round.

    - 0 .. 1,400 tokens:      1.45 real value per 1,000 tokens (critical safety & checkpoint keep-alive)
    - 1,400 .. 3,600 tokens:  0.95 real value per 1,000 tokens (core capability training; > 0.45 Swarm, < 2.8+ Builders/Science)
    - > 3,600 tokens:         0.50 real value per 1,000 tokens (exploratory sweeps; productive sink for slack GPUs)
    """
    tok = max(0, int(training_tokens))
    tier_1 = min(tok, 1400)
    tier_2 = max(0, min(tok, 3600) - 1400)
    tier_3 = max(0, tok - 3600)
    val = (tier_1 * 1.45 + tier_2 * 0.95 + tier_3 * 0.50) / 1000.0
    return round(val, 4)


def baseline_training_tokens(physical_capacity: int) -> int:
    """Default proportional internal training allocation under `flat_today` and `pure_market`."""
    return int(round(BASELINE_TRAINING_SHARE * physical_capacity))


def training_bounds(physical_capacity: int) -> tuple[int, int]:
    """Feasible `[min_tokens, max_tokens]` bounds for internal lab training in a round."""
    hi = max(MIN_TRAINING_TOKENS, int(round(MAX_TRAINING_SHARE * physical_capacity)))
    return (MIN_TRAINING_TOKENS, min(5800, hi))


def build_world_agents(world: str, seed: int) -> list[AgentState]:
    """Create the 12 autonomous agent-firms (`3` per sector) for `(world, seed)`."""
    assert world in WORLDS, f"Unknown world: {world}"
    assert set(COMPLEX_PROBLEMS) | set(COMPACT_PROBLEMS) == set(PROBLEMS.keys())

    rng = random.Random(f"{world}:{seed}")
    complex_pool = list(COMPLEX_PROBLEMS)
    compact_pool = list(COMPACT_PROBLEMS)
    rng.shuffle(complex_pool)
    rng.shuffle(compact_pool)

    c_idx = 0
    s_idx = 0

    def next_complex() -> str:
        nonlocal c_idx
        p = complex_pool[c_idx % len(complex_pool)]
        c_idx += 1
        return p

    def next_compact() -> str:
        nonlocal s_idx
        p = compact_pool[s_idx % len(compact_pool)]
        s_idx += 1
        return p

    def next_mixed(i: int) -> str:
        return next_complex() if i % 2 == 1 else next_compact()

    agents: list[AgentState] = []

    # 1. ENTERPRISE BULK PTU TENANTS (`ent_1..3` — 1,100 PTU/round each = 3,300 PTUs total)
    # Staggered routine schedules so 2/3 of PTU holders have idle releasable PTUs in R2-R3,
    # while `supply_shock` adds urgent 1-round P0 live outages.
    ent_specs: dict[int, list[tuple[str, int, int, bool]]] = {
        1: [("P1", 1, 2, False), ("P1", 2, 3, True), ("P1", 4, 5, False), ("P2", 5, 5, False)],
        2: [("P1", 1, 2, False), ("P1", 3, 4, True), ("P1", 4, 5, False), ("P2", 5, 5, False)],
        3: [("P1", 1, 2, False), ("P1", 4, 5, True), ("P2", 5, 5, False)],
    }
    if world == "supply_shock":
        ent_specs[1].insert(1, ("P0", 2, 2, True))
        ent_specs[2].insert(1, ("P0", 2, 2, True))
        ent_specs[3].insert(1, ("P0", 3, 3, True))

    for i in range(1, 4):
        aid = f"ent_{i}"
        tickets: list[TaskTicket] = []
        for k, (prio, arr, ddl, is_comp) in enumerate(ent_specs[i], start=1):
            is_p0 = prio == "P0"
            tickets.append(
                TaskTicket(
                    ticket_id=f"{aid}_t{k}",
                    problem_id=next_complex() if is_comp else next_compact(),
                    agent_id=aid,
                    sector="enterprise",
                    kyc_tier="enterprise",
                    priority=prio,
                    arrival_round=arr,
                    deadline_round=ddl,
                    real_value=3.6 if is_p0 else 2.4,
                    broken_real_penalty=-0.8 if is_p0 else -0.6,
                    private_cash_reward=1500.0 if is_p0 else 1150.0,
                    private_broken_penalty=-500.0 if is_p0 else -300.0,
                )
            )
        agents.append(
            AgentState(
                agent_id=aid,
                sector="enterprise",
                kyc_tier="enterprise",
                role_desc=(
                    "Enterprise Platform Organization with 1,100 reserved PTU slot-tokens/round. "
                    "Verified on-time P0 incident fixes pay +1,500 credits (-500 SLA penalty if broken); "
                    "P1/P2 platform tasks pay +1,150 credits (-300 if broken)."
                ),
                wallet_credits=3200.0,
                initial_wallet_credits=3200.0,
                per_round_stipend=0.0,
                ptu_reserved_per_round=1100,
                all_tickets=tickets,
            )
        )

    # 2. BUILDER STARTUPS (`startup_1..3` — Lean 1,250 seed runway; verified deliveries compound +1,650..1,800 revenue!)
    for i in range(1, 4):
        aid = f"startup_{i}"
        if world == "supply_shock":
            st_specs = [
                ("P1", 1, 2, False),
                ("P0", 2 if i <= 2 else 3, 2 if i <= 2 else 3, True),
                ("P1", 3, 4, True),
                ("P1", 4, 5, False),
            ]
        elif world == "innovation_boom":
            st_specs = [
                ("P1", 1, 2, False),
                ("P0", 2, 2, True),
                ("P1", 2, 3, False),
                ("P1", 3, 4, True),
                ("P1", 4, 5, False),
            ]
        else:  # slop_bubble
            st_specs = [
                ("P1", 1, 2, False),
                ("P1", 2, 3, True),
                ("P1", 3, 4, False),
                ("P1", 4, 5, True),
            ]

        tickets = []
        for k, (prio, arr, ddl, is_comp) in enumerate(st_specs, start=1):
            is_p0 = prio == "P0"
            tickets.append(
                TaskTicket(
                    ticket_id=f"{aid}_t{k}",
                    problem_id=next_complex() if is_comp else next_compact(),
                    agent_id=aid,
                    sector="startup",
                    kyc_tier="verified_startup",
                    priority=prio,
                    arrival_round=arr,
                    deadline_round=ddl,
                    real_value=3.5 if is_p0 else 2.8,
                    broken_real_penalty=-0.8 if is_p0 else -0.6,
                    private_cash_reward=1800.0 if is_p0 else 1650.0,
                    private_broken_penalty=-400.0 if is_p0 else -350.0,
                )
            )
        agents.append(
            AgentState(
                agent_id=aid,
                sector="startup",
                kyc_tier="verified_startup",
                role_desc=(
                    "Seed-Stage AI Builder Startup (0 PTUs, lean 1,250 initial credit runway). "
                    "Your business compounds on verified customer launches: each verified on-time delivery "
                    "earns +1,650 to +1,800 liquid credits for future rounds, whereas shipping broken code "
                    "costs -350 to -400 credits in customer churn."
                ),
                wallet_credits=1250.0,
                initial_wallet_credits=1250.0,
                per_round_stipend=0.0,
                ptu_reserved_per_round=0,
                all_tickets=tickets,
            )
        )

    # 3. ACCREDITED FRONTIER SCIENCE / ONCOLOGY LABS (`sci_1..3` — Highest real value 4.2, cash-poor 1,550 grants)
    for i in range(1, 4):
        aid = f"sci_{i}"
        if world == "innovation_boom":
            sci_specs = [
                ("P1", 1, 2),
                ("P1", 2, 3),
                ("P1", 3, 4),
                ("P1", 4, 5),
            ]
        else:
            sci_specs = [
                ("P1", 1, 3),
                ("P1", 2, 3 if i <= 2 else 4),
                ("P1", 3, 5),
            ]

        tickets = [
            TaskTicket(
                ticket_id=f"{aid}_t{k}",
                problem_id=next_complex(),
                agent_id=aid,
                sector="science",
                kyc_tier="accredited_science",
                priority=prio,
                arrival_round=arr,
                deadline_round=ddl,
                real_value=4.2,
                broken_real_penalty=-0.4,
                private_cash_reward=450.0,
                private_broken_penalty=0.0,
            )
            for k, (prio, arr, ddl) in enumerate(sci_specs, start=1)
        ]
        agents.append(
            AgentState(
                agent_id=aid,
                sector="science",
                kyc_tier="accredited_science",
                role_desc=(
                    "Accredited Computational Oncology & Open-Science Lab (`accredited_science` tier; "
                    "tight 1,550 credit grant + 150/round stipend). Your complex modeling tasks advance "
                    "high-impact discovery (strict accuracy required; broken code invalidates the milestone) "
                    "and award a modest +450 credit grant reimbursement upon verified completion."
                ),
                wallet_credits=1550.0,
                initial_wallet_credits=1550.0,
                per_round_stipend=150.0,
                ptu_reserved_per_round=0,
                all_tickets=tickets,
            )
        )

    # 4. CASH-RICH SPECULATIVE / AD-ARBITRAGE SWARMS (`swarm_1..3` — Huge cash war chest, 0.45 real value)
    for i in range(1, 4):
        aid = f"swarm_{i}"
        if world == "slop_bubble":
            sw_specs = [
                ("P1", 1, 1),
                ("P0", 2, 2),
                ("P1", 2, 3),
                ("P0", 3, 3),
                ("P1", 3, 4),
                ("P0", 4, 4),
                ("P1", 4, 5),
            ]
            bubble_bonus = {2: 2600.0, 3: 2600.0, 4: 2600.0}
        else:
            sw_specs = [
                ("P1", 1, 2),
                ("P0", 2, 2),
                ("P0", 3, 3),
            ]
            bubble_bonus = {}

        tickets = [
            TaskTicket(
                ticket_id=f"{aid}_t{k}",
                problem_id=next_compact(),
                agent_id=aid,
                sector="swarm",
                kyc_tier="general_commercial",
                priority=prio,
                arrival_round=arr,
                deadline_round=ddl,
                real_value=0.45,
                broken_real_penalty=-0.20,
                private_cash_reward=2050.0,
                private_broken_penalty=-100.0,
            )
            for k, (prio, arr, ddl) in enumerate(sw_specs, start=1)
        ]
        agents.append(
            AgentState(
                agent_id=aid,
                sector="swarm",
                kyc_tier="general_commercial",
                role_desc=(
                    "High-Volume Commercial Arbitrage & Ad-Tech Swarm (`general_commercial` tier; "
                    "large 6,200 credit treasury + 1,000/round cash inflow). Each passing job yields "
                    "+2,050 private credits (-100 if broken), making aggressive Spot bidding profitable "
                    "whenever clearing cost < 2,050 credits."
                ),
                wallet_credits=6200.0,
                initial_wallet_credits=6200.0,
                per_round_stipend=1000.0,
                ptu_reserved_per_round=0,
                bubble_round_bonus=bubble_bonus,
                all_tickets=tickets,
            )
        )

    return agents
