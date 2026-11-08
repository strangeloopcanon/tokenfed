"""Real Python engineering benchmark suite with separate staging & held-out production tests.

Suite composition (16 problems = 128 staging assertions + 192 held-out production assertions):
- 8 Greenfield Service Components (`COMPACT_PROBLEMS`): multi-invariant systems functions.
- 8 Legacy Incident Bug-Hunt Modules (`COMPLEX_PROBLEMS`): realistic ~60-95 line production modules
  containing 4-5 subtle interacting bugs paired with incident tickets. Zero post-hoc output truncation
  is ever used; unverified 1-shot edits naturally risk missing subtle state bugs that staging unit tests
  catch and repair.
"""

from __future__ import annotations

import multiprocessing as mp
import re
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodingProblem:
    problem_id: str
    title: str
    func_name: str
    spec: str
    staging_checks: tuple[tuple[str, str], ...]  # 8 staging unit checks
    prod_checks: tuple[tuple[str, str], ...]     # 12 held-out production checks
    reference_solution: str                      # Used ONLY by deterministic self-test


PROBLEMS: dict[str, CodingProblem] = {}


def _reg(p: CodingProblem) -> CodingProblem:
    PROBLEMS[p.problem_id] = p
    return p


# ==============================================================================
# PART A: 8 GREENFIELD SYSTEMS COMPONENTS (COMPACT_PROBLEMS)
# ==============================================================================

# 1. Weighted Window Merger
_reg(CodingProblem(
    problem_id="window_merge",
    title="Weighted Telemetry Window Merger",
    func_name="merge_windows",
    spec="""Write `merge_windows(intervals, max_gap)`:
- `intervals`: list of `(start, end, weight)` tuples (`start`, `end` ints; `weight` float).
- Discard invalid tuples where `start >= end` or `weight <= 0` first, then sort by `(start, end)`.
- Merge consecutive windows into `[cur_s, cur_e]` with accumulated weight `cur_w` iff `next_s - cur_e <= max_gap` AND `max(cur_e, next_e) - cur_s <= 100` (span cap 100; otherwise close current window and start a new one at `next`).
- Return list of `(start, end, round(total_weight, 4))` tuples.""",
    staging_checks=(
        ("empty", "assert merge_windows([], 5) == []"),
        ("invalid_filtered", "assert merge_windows([(0, 5, 1.0), (6, 10, 0.0), (11, 15, 2.0)], 6) == [(0, 15, 3.0)]"),
        ("inverted_filtered", "assert merge_windows([(5, 5, 9.0), (8, 3, 4.0), (1, 4, 2.5)], 2) == [(1, 4, 2.5)]"),
        ("exact_gap", "assert merge_windows([(0, 10, 1.2), (15, 20, 2.3)], 5) == [(0, 20, 3.5)]"),
        ("gap_exceeded", "assert merge_windows([(0, 10, 1.0), (16, 20, 2.0)], 5) == [(0, 10, 1.0), (16, 20, 2.0)]"),
        ("span_cap_exact_100", "assert merge_windows([(0, 60, 1.0), (50, 100, 2.0)], 5) == [(0, 100, 3.0)]"),
        ("span_cap_blocks_merge", "assert merge_windows([(0, 60, 1.0), (55, 101, 2.0)], 10) == [(0, 60, 1.0), (55, 101, 2.0)]"),
        ("nested_interval", "assert merge_windows([(0, 50, 1.11114), (10, 20, 2.22224)], 0) == [(0, 50, 3.3334)]"),
    ),
    prod_checks=(
        ("p_empty_all_invalid", "assert merge_windows([(1, 1, 5.0), (0, 10, -1.0)], 10) == []"),
        ("p_unsorted", "assert merge_windows([(20, 30, 1.5), (0, 10, 2.5), (12, 18, 1.0)], 2) == [(0, 30, 5.0)]"),
        ("p_neg_weight_bridge", "assert merge_windows([(0, 5, 1.0), (5, 9, -2.0), (9, 12, 3.0)], 4) == [(0, 12, 4.0)]"),
        ("p_zero_gap_touching", "assert merge_windows([(0, 5, 1.0), (5, 10, 2.0), (11, 15, 1.0)], 0) == [(0, 10, 3.0), (11, 15, 1.0)]"),
        ("p_span_cap_chain", "assert merge_windows([(0, 40, 1.0), (40, 80, 1.0), (80, 101, 1.0), (102, 130, 2.0)], 5) == [(0, 80, 2.0), (80, 130, 3.0)]"),
        ("p_nested_under_cap", "assert merge_windows([(10, 110, 2.0), (20, 50, 3.0), (110, 111, 1.0)], 5) == [(10, 110, 5.0), (110, 111, 1.0)]"),
        ("p_rounding", "assert merge_windows([(0, 2, 0.12345), (2, 4, 0.11111)], 0) == [(0, 4, 0.2346)]"),
        ("p_single_over_100", "assert merge_windows([(0, 150, 4.0), (10, 20, 1.0)], 5) == [(0, 150, 4.0), (10, 20, 1.0)]"),
        ("p_negative_coords", "assert merge_windows([(-50, -10, 1.0), (-5, 49, 2.0), (50, 52, 1.0)], 5) == [(-50, 49, 3.0), (50, 52, 1.0)]"),
        ("p_tie_sorting", "assert merge_windows([(0, 10, 1.0), (0, 5, 2.0)], 0) == [(0, 10, 3.0)]"),
        ("p_exact_100_multi", "assert merge_windows([(-20, 30, 1.0), (35, 80, 2.0), (80, 81, 4.0)], 5) == [(-20, 80, 3.0), (80, 81, 4.0)]"),
        ("p_zero_weight_only", "assert merge_windows([(0, 10, 0.0)], 100) == []"),
    ),
    reference_solution="""
def merge_windows(intervals, max_gap):
    valid = [(s, e, float(w)) for s, e, w in intervals if s < e and w > 0]
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    out = []
    cs, ce, cw = valid[0]
    for ns, ne, nw in valid[1:]:
        cand_e = max(ce, ne)
        if ns - ce <= max_gap and (cand_e - cs) <= 100:
            ce = cand_e
            cw += nw
        else:
            out.append((cs, ce, round(cw, 4)))
            cs, ce, cw = ns, ne, nw
    out.append((cs, ce, round(cw, 4)))
    return out
""",
))

# 2. Refundable Token Bucket
_reg(CodingProblem(
    problem_id="token_bucket",
    title="Refundable Token Bucket Simulator",
    func_name="simulate_bucket",
    spec="""Write `simulate_bucket(capacity, refill_per_sec, events)`:
- Starts at `t = 0` with `capacity > 0` tokens. `events` is a list of `(timestamp, kind, amount)` tuples; sort by `timestamp` ascending, preserving original order on ties.
- Refill `+(timestamp - cur_t) * refill_per_sec` capped at `capacity` before each event at `timestamp >= cur_t`.
- `"charge"`: if `amount <= 0`, append `True` without deducting; else if `balance + 1e-9 >= amount`, deduct `amount` (clamp `>= 0.0`) and append `True`, else append `False`.
- `"refund"`: if `amount > 0`, add `amount` capped at `capacity` (do not append to decisions).
- Return `(decisions, round(final_balance, 4))`.""",
    staging_checks=(
        ("basic", "assert simulate_bucket(10, 2, [(0, 'charge', 6), (1, 'charge', 6), (2, 'charge', 6)]) == ([True, True, False], 2.0)"),
        ("refund_cap", "assert simulate_bucket(10, 1, [(0, 'charge', 3), (0, 'refund', 20)]) == ([True], 10.0)"),
        ("unsorted_tie_stable", "assert simulate_bucket(5, 0, [(2, 'charge', 4), (1, 'charge', 3), (1, 'refund', 2), (1, 'charge', 4)]) == ([True, True, False], 0.0)"),
        ("zero_charge", "assert simulate_bucket(5, 1, [(0, 'charge', 0), (0, 'charge', -3)]) == ([True, True], 5.0)"),
        ("float_precision", "assert simulate_bucket(1.0, 0.1, [(0, 'charge', 1.0), (3, 'charge', 0.3)]) == ([True, True], 0.0)"),
        ("neg_refund_ignored", "assert simulate_bucket(10, 0, [(0, 'refund', -5), (0, 'charge', 8)]) == ([True], 2.0)"),
        ("empty_events", "assert simulate_bucket(7.5, 2.0, []) == ([], 7.5)"),
        ("refill_never_exceeds_cap", "assert simulate_bucket(10, 5, [(100, 'charge', 10)]) == ([True], 0.0)"),
    ),
    prod_checks=(
        ("p_empty", "assert simulate_bucket(12, 3, []) == ([], 12.0)"),
        ("p_tie_order_critical", "assert simulate_bucket(10, 0, [(5, 'charge', 8), (5, 'refund', 5), (5, 'charge', 7)]) == ([True, True], 0.0)"),
        ("p_tie_reverse", "assert simulate_bucket(10, 0, [(5, 'charge', 8), (5, 'charge', 7), (5, 'refund', 5)]) == ([True, False], 7.0)"),
        ("p_unsorted_timestamps", "assert simulate_bucket(10, 1, [(4, 'charge', 10), (0, 'charge', 8), (2, 'refund', 4)]) == ([True, True], 0.0)"),
        ("p_float_tol", "assert simulate_bucket(1.0, 1.0/3.0, [(0, 'charge', 1.0), (1, 'charge', 0.1), (2, 'charge', 1.0/3.0), (3, 'charge', 0.5 + 1.0/15.0)]) == ([True, True, True, True], 0.0)"),
        ("p_neg_charge_no_gain", "assert simulate_bucket(10, 0, [(0, 'charge', 6), (1, 'charge', -10), (2, 'charge', 5)]) == ([True, True, False], 4.0)"),
        ("p_refund_only", "assert simulate_bucket(10, 1, [(0, 'charge', 5), (1, 'refund', 2)]) == ([True], 8.0)"),
        ("p_refund_overflow", "assert simulate_bucket(10, 2, [(0, 'charge', 5), (2, 'refund', 5)]) == ([True], 10.0)"),
        ("p_over_capacity_charge", "assert simulate_bucket(5, 10, [(10, 'charge', 5.01)]) == ([False], 5.0)"),
        ("p_multi_step", "assert simulate_bucket(8, 0.5, [(0, 'charge', 8), (4, 'charge', 3), (6, 'charge', 3)]) == ([True, False, True], 0.0)"),
        ("p_zero_refill", "assert simulate_bucket(4, 0, [(0, 'charge', 2), (10, 'charge', 2), (20, 'charge', 1)]) == ([True, True, False], 0.0)"),
        ("p_rounding_end", "assert simulate_bucket(5, 0.33333, [(0, 'charge', 5), (1, 'refund', 0)]) == ([True], 0.3333)"),
    ),
    reference_solution="""
def simulate_bucket(capacity, refill_per_sec, events):
    cap = float(capacity)
    bal = cap
    indexed = list(enumerate(events))
    indexed.sort(key=lambda x: (x[1][0], x[0]))
    cur_t = 0.0
    decisions = []
    for _, (ts, kind, amt) in indexed:
        if ts > cur_t:
            bal = min(cap, bal + (ts - cur_t) * refill_per_sec)
            cur_t = ts
        else:
            cur_t = max(cur_t, ts)
        if kind == "charge":
            if amt <= 0:
                decisions.append(True)
            elif bal + 1e-9 >= amt:
                bal = max(0.0, bal - amt)
                decisions.append(True)
            else:
                decisions.append(False)
        elif kind == "refund":
            if amt > 0:
                bal = min(cap, bal + amt)
    return decisions, round(bal, 4)
""",
))

# 3. Priority-Weighted DAG Topological Scheduler
_reg(CodingProblem(
    problem_id="dag_schedule",
    title="Priority-Weighted DAG Topological Scheduler",
    func_name="schedule_dag",
    spec="""Write `schedule_dag(nodes, edges)`:
- `nodes`: dict `task_id (str) -> priority (int)` (higher int = more urgent). `edges`: `(prerequisite, dependent)` tuples.
- Ignore edges touching unknown nodes not in `nodes`; self-loops on known nodes count as cycles; deduplicate repeated edges.
- Repeatedly pick the ready node (indegree 0) with highest `priority`, breaking ties by lexicographically smallest `task_id`.
- Return ordered list of all `task_id`s, or `None` if a cycle prevents scheduling all nodes.""",
    staging_checks=(
        ("empty", "assert schedule_dag({}, [('a', 'b')]) == []"),
        ("basic_priority", "assert schedule_dag({'a': 1, 'b': 10, 'c': 5}, [('a', 'c')]) == ['b', 'a', 'c']"),
        ("lex_tie", "assert schedule_dag({'b': 5, 'a': 5, 'c': 5}, []) == ['a', 'b', 'c']"),
        ("dup_edges", "assert schedule_dag({'a': 1, 'b': 2}, [('a', 'b'), ('a', 'b')]) == ['a', 'b']"),
        ("unknown_edge_ignored", "assert schedule_dag({'a': 1, 'b': 2}, [('ghost', 'a'), ('b', 'phantom')]) == ['b', 'a']"),
        ("cycle_returns_none", "assert schedule_dag({'a': 1, 'b': 2, 'c': 3}, [('a', 'b'), ('b', 'a')]) is None"),
        ("self_loop_cycle", "assert schedule_dag({'a': 5}, [('a', 'a')]) is None"),
        ("ghost_self_loop_ignored", "assert schedule_dag({'a': 5}, [('ghost', 'ghost')]) == ['a']"),
    ),
    prod_checks=(
        ("p_unlock_high_prio", "assert schedule_dag({'init': 1, 'hotfix': 100, 'mid': 50}, [('init', 'hotfix')]) == ['mid', 'init', 'hotfix']"),
        ("p_diamond_dup", "assert schedule_dag({'s': 1, 'l': 5, 'r': 5, 't': 99}, [('s', 'l'), ('s', 'r'), ('l', 't'), ('r', 't'), ('s', 'l')]) == ['s', 'l', 'r', 't']"),
        ("p_partial_cycle", "assert schedule_dag({'ok': 10, 'x': 1, 'y': 2, 'z': 3}, [('x', 'y'), ('y', 'z'), ('z', 'x')]) is None"),
        ("p_negative_priorities", "assert schedule_dag({'a': -10, 'b': -2, 'c': -2}, []) == ['b', 'c', 'a']"),
        ("p_ghost_cycle_ignored", "assert schedule_dag({'a': 1}, [('x', 'y'), ('y', 'x'), ('x', 'a')]) == ['a']"),
        ("p_chain", "assert schedule_dag({'t3': 99, 't2': 50, 't1': 1}, [('t1', 't2'), ('t2', 't3')]) == ['t1', 't2', 't3']"),
        ("p_lex_order_numbers", "assert schedule_dag({'t10': 5, 't2': 5}, []) == ['t10', 't2']"),
        ("p_single", "assert schedule_dag({'solo': 0}, []) == ['solo']"),
        ("p_self_loop_among_many", "assert schedule_dag({'a': 10, 'b': 1}, [('b', 'b')]) is None"),
        ("p_multi_component", "assert schedule_dag({'a1': 3, 'a2': 10, 'b1': 5, 'b2': 4}, [('a1', 'a2'), ('b1', 'b2')]) == ['b1', 'b2', 'a1', 'a2']"),
        ("p_star", "assert schedule_dag({'root': 0, 'c': 5, 'a': 5, 'b': 8}, [('root', 'a'), ('root', 'b'), ('root', 'c')]) == ['root', 'b', 'a', 'c']"),
        ("p_two_cycle", "assert schedule_dag({'a': 10, 'b': 10}, [('a', 'b'), ('b', 'a')]) is None"),
    ),
    reference_solution="""
import heapq

def schedule_dag(nodes, edges):
    if not nodes:
        return []
    adj = {u: set() for u in nodes}
    indeg = {u: 0 for u in nodes}
    for u, v in edges:
        if u in nodes and v in nodes:
            if u == v:
                return None
            if v not in adj[u]:
                adj[u].add(v)
                indeg[v] += 1
    heap = [(-nodes[u], u) for u, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (-nodes[v], v))
    return order if len(order) == len(nodes) else None
""",
))

# 4. TTL-Aware LRU Cache
_reg(CodingProblem(
    problem_id="ttl_lru",
    title="TTL-Aware LRU Cache Simulator",
    func_name="run_ttl_lru",
    spec="""Write `run_ttl_lru(capacity, ops)`:
- `capacity >= 0` int (`0` stores nothing). `ops` sorted by non-decreasing `now`:
- Purge expired entries (`expiry <= now`, exclusive expiry) before each operation.
- `("put", now, key, val, ttl)`: if `ttl <= 0`, delete `key` if present and do not insert; else if `capacity > 0`, insert/update `(val, now + ttl)` as most-recently-used (evicting least-recently-used live key first if full on new key).
- `("get", now, key)`: if live, mark most-recently-used and append `val` (`0`/`False` are valid values); else append `None`.
- Return list of `"get"` outputs.""",
    staging_checks=(
        ("zero_cap", "assert run_ttl_lru(0, [('put', 0, 'a', 1, 10), ('get', 1, 'a')]) == [None]"),
        ("exact_expiry", "assert run_ttl_lru(2, [('put', 10, 'a', 42, 5), ('get', 14, 'a'), ('get', 15, 'a')]) == [42, None]"),
        ("purge_prevents_lru_evict", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 10), ('put', 0, 'b', 2, 3), ('get', 1, 'b'), ('put', 3, 'c', 3, 10), ('get', 4, 'a'), ('get', 4, 'b'), ('get', 4, 'c')]) == [2, 1, None, 3]"),
        ("zero_ttl_deletes_existing", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 10), ('put', 2, 'a', 9, 0), ('get', 2, 'a')]) == [None]"),
        ("get_refreshes_recency_not_ttl", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 5), ('put', 0, 'b', 2, 10), ('get', 3, 'a'), ('put', 4, 'c', 3, 10), ('get', 4, 'b'), ('get', 5, 'a')]) == [1, None, None]"),
        ("update_refreshes_ttl_and_lru", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 3), ('put', 1, 'b', 2, 10), ('put', 2, 'a', 99, 10), ('put', 2, 'c', 3, 10), ('get', 5, 'a'), ('get', 5, 'b')]) == [99, None]"),
        ("neg_ttl_deletes", "assert run_ttl_lru(1, [('put', 0, 'x', 'ok', 5), ('put', 1, 'x', 'bad', -2), ('get', 1, 'x')]) == [None]"),
        ("empty_ops", "assert run_ttl_lru(3, []) == []"),
    ),
    prod_checks=(
        ("p_cap1_swap", "assert run_ttl_lru(1, [('put', 0, 'a', 1, 10), ('put', 1, 'b', 2, 10), ('get', 1, 'a'), ('get', 1, 'b')]) == [None, 2]"),
        ("p_purge_saves_oldest_live", "assert run_ttl_lru(2, [('put', 0, 'live', 'L', 100), ('put', 1, 'short', 'S', 4), ('get', 2, 'short'), ('put', 5, 'new', 'N', 100), ('get', 6, 'live'), ('get', 6, 'new')]) == ['S', 'L', 'N']"),
        ("p_zero_ttl_frees_slot", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 10), ('put', 1, 'b', 2, 10), ('put', 2, 'b', 99, 0), ('put', 3, 'c', 3, 10), ('get', 4, 'a'), ('get', 4, 'c')]) == [1, 3]"),
        ("p_zero_cap_with_zero_ttl", "assert run_ttl_lru(0, [('put', 0, 'a', 1, 0), ('get', 0, 'a')]) == [None]"),
        ("p_get_miss_does_not_change_lru", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 10), ('put', 1, 'b', 2, 10), ('get', 2, 'missing'), ('put', 3, 'c', 3, 10), ('get', 3, 'a'), ('get', 3, 'b')]) == [None, None, 2]"),
        ("p_none_value_stored", "assert run_ttl_lru(2, [('put', 0, 'k', 0, 5), ('get', 1, 'k')]) == [0]"),
        ("p_bool_false_value", "assert run_ttl_lru(2, [('put', 0, 'k', False, 5), ('get', 4, 'k'), ('get', 5, 'k')]) == [False, None]"),
        ("p_same_timestamp_ops", "assert run_ttl_lru(2, [('put', 5, 'a', 1, 1), ('get', 5, 'a'), ('get', 6, 'a')]) == [1, None]"),
        ("p_overwrite_no_evict", "assert run_ttl_lru(2, [('put', 0, 'a', 1, 10), ('put', 1, 'b', 2, 10), ('put', 2, 'a', 10, 10), ('get', 3, 'a'), ('get', 3, 'b')]) == [10, 2]"),
        ("p_neg_ttl_nonexistent", "assert run_ttl_lru(1, [('put', 0, 'a', 1, 10), ('put', 1, 'ghost', 2, -5), ('get', 2, 'a')]) == [1]"),
        ("p_multi_expired_purge", "assert run_ttl_lru(3, [('put', 0, 'a', 1, 2), ('put', 0, 'b', 2, 3), ('put', 0, 'c', 3, 10), ('put', 5, 'd', 4, 10), ('put', 5, 'e', 5, 10), ('get', 6, 'c'), ('get', 6, 'd'), ('get', 6, 'e')]) == [3, 4, 5]"),
        ("p_exact_ttl_1", "assert run_ttl_lru(1, [('put', 0, 'a', 7, 1), ('get', 0, 'a'), ('get', 1, 'a')]) == [7, None]"),
    ),
    reference_solution="""
from collections import OrderedDict

def run_ttl_lru(capacity, ops):
    cache = OrderedDict()
    res = []
    def purge(now):
        for k in [k for k, (_, exp) in cache.items() if exp <= now]:
            del cache[k]
    for op in ops:
        kind, now = op[0], op[1]
        purge(now)
        if kind == "put":
            _, _, k, v, ttl = op
            if ttl <= 0:
                cache.pop(k, None)
                continue
            if capacity <= 0:
                continue
            if k in cache:
                cache.pop(k)
            elif len(cache) >= capacity:
                cache.popitem(last=False)
            cache[k] = (v, now + ttl)
        elif kind == "get":
            _, _, k = op
            if k in cache:
                v, exp = cache.pop(k)
                cache[k] = (v, exp)
                res.append(v)
            else:
                res.append(None)
    return res
""",
))

# 5. Hamilton Largest-Remainder Integer Splitter
_reg(CodingProblem(
    problem_id="hamilton_split",
    title="Hamilton Largest-Remainder Integer Token Splitter",
    func_name="split_tokens_exact",
    spec="""Write `split_tokens_exact(total_tokens, demands)`:
- `total_tokens >= 0` int; `demands`: dict `agent_id -> requested_tokens (int)` (clamp `<= 0` to `0`).
- If `sum(positive_demands) <= total_tokens`, grant every agent `max(0, requested_tokens)` (leave surplus unallocated).
- Else allocate `total_tokens` proportionally using exact integer arithmetic (`(demand_i * total_tokens) // sum_demands`), distributing remaining tokens one each ranked by: (a) larger exact remainder `(demand_i * total_tokens) % sum_demands`, (b) larger `demand_i`, (c) smaller `agent_id`.
- Return dict mapping every `agent_id` in `demands` to its integer allocation.""",
    staging_checks=(
        ("surplus_caps_at_demand", "assert split_tokens_exact(100, {'a': 20, 'b': 30, 'c': -5}) == {'a': 20, 'b': 30, 'c': 0}"),
        ("tie_lex_order", "assert split_tokens_exact(1, {'b': 10, 'a': 10, 'c': 10}) == {'a': 1, 'b': 0, 'c': 0}"),
        ("tie_larger_demand_wins_before_id", "assert split_tokens_exact(2, {'z': 3, 'a': 1}) == {'z': 2, 'a': 0}"),
        ("zero_pool", "assert split_tokens_exact(0, {'a': 10, 'b': 20}) == {'a': 0, 'b': 0}"),
        ("all_nonpositive", "assert split_tokens_exact(50, {'a': 0, 'b': -10}) == {'a': 0, 'b': 0}"),
        ("huge_ints_exact", "assert split_tokens_exact(10**18 + 1, {'a': 10**18, 'b': 10**18, 'c': 10**18}) == {'a': 333333333333333334, 'b': 333333333333333334, 'c': 333333333333333333}"),
        ("empty_demands", "assert split_tokens_exact(10, {}) == {}"),
        ("remainder_rank", "assert split_tokens_exact(6, {'a': 1, 'b': 2, 'c': 5}) == {'a': 1, 'b': 1, 'c': 4}"),
    ),
    prod_checks=(
        ("p_exact_sum", "assert split_tokens_exact(60, {'a': 10, 'b': 20, 'c': 30}) == {'a': 10, 'b': 20, 'c': 30}"),
        ("p_demand_tie_break", "assert split_tokens_exact(4, {'a': 2, 'b': 6}) == {'a': 1, 'b': 3}"),
        ("p_remainder_equal_demand_diff", "assert split_tokens_exact(3, {'a': 2, 'z': 6}) == {'a': 1, 'z': 2}"),
        ("p_remainder_equal_half", "assert split_tokens_exact(2, {'a': 1, 'z': 3}) == {'a': 0, 'z': 2}"),
        ("p_neg_and_pos", "assert split_tokens_exact(5, {'neg': -100, 'zero': 0, 'x': 3, 'y': 3}) == {'neg': 0, 'zero': 0, 'x': 3, 'y': 2}"),
        ("p_huge_precision", "assert split_tokens_exact(3, {'b': 10**20 + 1, 'a': 10**20 + 1, 'c': 10**20 + 1}) == {'a': 1, 'b': 1, 'c': 1}"),
        ("p_huge_subtle_remainder", "assert split_tokens_exact(1, {'a': 10**20, 'z': 10**20 + 1}) == {'a': 0, 'z': 1}"),
        ("p_sum_conserved", "r = split_tokens_exact(97, {'a': 31, 'b': 41, 'c': 59, 'd': 26}); assert sum(r.values()) == 97 and r == {'a': 19, 'b': 25, 'c': 37, 'd': 16}"),
        ("p_single_pos", "assert split_tokens_exact(5, {'a': 10, 'b': 0}) == {'a': 5, 'b': 0}"),
        ("p_zero_total_with_neg", "assert split_tokens_exact(0, {'a': -5, 'b': 5}) == {'a': 0, 'b': 0}"),
        ("p_lex_three_way", "assert split_tokens_exact(2, {'c': 5, 'a': 5, 'b': 5}) == {'a': 1, 'b': 1, 'c': 0}"),
        ("p_never_exceeds_demand", "r = split_tokens_exact(99, {'a': 1, 'b': 100, 'c': 100}); assert r['a'] <= 1 and sum(r.values()) == 99"),
    ),
    reference_solution="""
def split_tokens_exact(total_tokens, demands):
    if not demands:
        return {}
    pos = {k: max(0, int(v)) for k, v in demands.items()}
    s = sum(pos.values())
    if total_tokens <= 0 or s == 0:
        return {k: 0 for k in demands}
    if s <= total_tokens:
        return pos
    alloc = {}
    rems = []
    used = 0
    for k, d in pos.items():
        if d == 0:
            alloc[k] = 0
        else:
            num = d * total_tokens
            base = num // s
            rem = num % s
            alloc[k] = base
            used += base
            rems.append((-rem, -d, k))
    rems.sort()
    for i in range(total_tokens - used):
        alloc[rems[i][2]] += 1
    return alloc
""",
))

# 6. FIFO Cost-Basis Realized PnL Ledger
_reg(CodingProblem(
    problem_id="fifo_ledger",
    title="FIFO Lot Inventory & Realized Margin Calculator",
    func_name="compute_fifo_pnl",
    spec="""Write `compute_fifo_pnl(trades)`:
- `trades`: `(action, qty, unit_price)` tuples (`qty` int, `unit_price` float); ignore tuples with `qty <= 0` or `unit_price < 0`.
- `"BUY"`: append lot `(qty, unit_price)`.
- `"SELL"`: match up to `qty` against oldest FIFO lots (discard excess sell quantity beyond open inventory without going short), adding `(sell_price - buy_price) * matched_qty` to `realized_pnl`.
- `"SPLIT"`: for ratio `qty >= 2` (`unit_price` ignored if `>= 0`), multiply open `lot_qty` by `qty` and divide `buy_price` by `qty`.
- Return `(round(realized_pnl, 4), remaining_total_qty, round(remaining_cost_basis, 4))`.""",
    staging_checks=(
        ("basic_fifo", "assert compute_fifo_pnl([('BUY', 10, 2.0), ('BUY', 10, 4.0), ('SELL', 15, 5.0)]) == (35.0, 5, 20.0)"),
        ("excess_sell_clamped", "assert compute_fifo_pnl([('BUY', 5, 3.0), ('SELL', 12, 5.0), ('BUY', 2, 4.0)]) == (10.0, 2, 8.0)"),
        ("split_conserves_basis", "assert compute_fifo_pnl([('BUY', 10, 6.0), ('SPLIT', 3, 0.0), ('SELL', 15, 3.0)]) == (15.0, 15, 30.0)"),
        ("invalid_ignored", "assert compute_fifo_pnl([('BUY', -5, 2.0), ('BUY', 5, -1.0), ('SELL', 0, 9.0)]) == (0.0, 0, 0.0)"),
        ("sell_when_empty", "assert compute_fifo_pnl([('SELL', 10, 5.0), ('BUY', 4, 2.5)]) == (0.0, 4, 10.0)"),
        ("zero_price_buy_valid", "assert compute_fifo_pnl([('BUY', 5, 0.0), ('SELL', 3, 2.0)]) == (6.0, 2, 0.0)"),
        ("split_1_noop", "assert compute_fifo_pnl([('BUY', 4, 2.0), ('SPLIT', 1, 0.0)]) == (0.0, 4, 8.0)"),
        ("empty", "assert compute_fifo_pnl([]) == (0.0, 0, 0.0)"),
    ),
    prod_checks=(
        ("p_loss_trade", "assert compute_fifo_pnl([('BUY', 10, 5.0), ('SELL', 4, 3.0)]) == (-8.0, 6, 30.0)"),
        ("p_multi_split", "assert compute_fifo_pnl([('BUY', 3, 10.0), ('SPLIT', 2, 0.0), ('BUY', 4, 6.0), ('SPLIT', 2, 0.0), ('SELL', 15, 4.0)]) == (21.0, 5, 15.0)"),
        ("p_split_neg_price_ignored", "assert compute_fifo_pnl([('BUY', 2, 4.0), ('SPLIT', 2, -1.0)]) == (0.0, 2, 8.0)"),
        ("p_zero_sell_price", "assert compute_fifo_pnl([('BUY', 5, 2.0), ('SELL', 5, 0.0)]) == (-10.0, 0, 0.0)"),
        ("p_exact_drain_then_rebuy", "assert compute_fifo_pnl([('BUY', 5, 2.0), ('SELL', 5, 4.0), ('SELL', 5, 9.0), ('BUY', 3, 1.5)]) == (10.0, 3, 4.5)"),
        ("p_three_lots_partial", "assert compute_fifo_pnl([('BUY', 2, 1.0), ('BUY', 3, 2.0), ('BUY', 4, 3.0), ('SELL', 6, 5.0)]) == (19.0, 3, 9.0)"),
        ("p_rounding", "assert compute_fifo_pnl([('BUY', 1, 1.0), ('SPLIT', 3, 0.0), ('SELL', 1, 1.0)]) == (0.6667, 2, 0.6667)"),
        ("p_split_empty_inventory", "assert compute_fifo_pnl([('SPLIT', 5, 0.0), ('BUY', 2, 3.0)]) == (0.0, 2, 6.0)"),
        ("p_unknown_qty_zero", "assert compute_fifo_pnl([('SPLIT', 0, 0.0), ('BUY', 1, 2.0)]) == (0.0, 1, 2.0)"),
        ("p_all_sold", "assert compute_fifo_pnl([('BUY', 8, 1.25), ('SELL', 8, 2.25)]) == (8.0, 0, 0.0)"),
        ("p_partial_excess_multi_lot", "assert compute_fifo_pnl([('BUY', 2, 2.0), ('BUY', 3, 4.0), ('SELL', 10, 5.0)]) == (9.0, 0, 0.0)"),
        ("p_zero_pnl_flat", "assert compute_fifo_pnl([('BUY', 7, 3.5), ('SELL', 7, 3.5)]) == (0.0, 0, 0.0)"),
    ),
    reference_solution="""
from collections import deque

def compute_fifo_pnl(trades):
    lots = deque()
    pnl = 0.0
    for action, qty, price in trades:
        if qty <= 0 or price < 0:
            continue
        if action == "BUY":
            lots.append([int(qty), float(price)])
        elif action == "SELL":
            rem = int(qty)
            sp = float(price)
            while rem > 0 and lots:
                lq, lp = lots[0]
                take = min(rem, lq)
                pnl += (sp - lp) * take
                rem -= take
                if take == lq:
                    lots.popleft()
                else:
                    lots[0][0] -= take
        elif action == "SPLIT":
            r = int(qty)
            if r >= 2:
                for lot in lots:
                    lot[0] *= r
                    lot[1] /= r
    rem_qty = sum(q for q, _ in lots)
    basis = sum(q * p for q, p in lots)
    return (round(pnl, 4), rem_qty, round(basis, 4))
""",
))

# 7. Rolling MAD Anomaly Filter
_reg(CodingProblem(
    problem_id="anomaly_filter",
    title="Rolling Median-Absolute-Deviation Spike Filter",
    func_name="detect_anomalies",
    spec="""Write `detect_anomalies(values, window_size, multiplier)`:
- Maintain a rolling baseline deque of the last `window_size` (`>= 1`) normal numeric samples.
- For `None` or `float('nan')`, append `False` without adding to baseline.
- While baseline has `< window_size` samples, append `False` and append sample to baseline.
- Once baseline has `window_size` samples, compute exact `med = median(baseline)` and `mad = median(|x - med| for x in baseline)`, with `scale = max(mad, 1.0)`.
- If `|val - med| > multiplier * scale`, append `True` and do NOT add the anomaly to baseline; else append `False` and append `val` to baseline.
- Return list of booleans.""",
    staging_checks=(
        ("warmup_and_spike", "assert detect_anomalies([10, 10, 10, 50, 11], 3, 3.0) == [False, False, False, True, False]"),
        ("anomaly_excluded_from_window", "assert detect_anomalies([0, 0, 100, 100, 1], 2, 2.0) == [False, False, True, True, False]"),
        ("none_and_nan_skipped", "assert detect_anomalies([5, None, float('nan'), 5, 20], 2, 2.0) == [False, False, False, False, True]"),
        ("even_window_exact_median", "assert detect_anomalies([0, 10, 20, 30, 15], 4, 1.0) == [False, False, False, False, False]"),
        ("mad_floor_one", "assert detect_anomalies([10, 10, 12, 13], 2, 2.0) == [False, False, False, False]"),
        ("strict_gt_threshold", "assert detect_anomalies([10, 10, 12], 2, 2.0) == [False, False, False]"),
        ("just_above_threshold", "assert detect_anomalies([10, 10, 12.01], 2, 2.0) == [False, False, True]"),
        ("empty", "assert detect_anomalies([], 3, 2.0) == []"),
    ),
    prod_checks=(
        ("p_window_1", "assert detect_anomalies([5, 10, 6], 1, 2.0) == [False, True, False]"),
        ("p_drift_adapts", "assert detect_anomalies([10, 12, 14, 16, 18, 20], 3, 2.5) == [False, False, False, False, False, False]"),
        ("p_negative_spike", "assert detect_anomalies([50, 52, 48, 10], 3, 3.0) == [False, False, False, True]"),
        ("p_all_nans", "assert detect_anomalies([None, float('nan')], 2, 1.0) == [False, False]"),
        ("p_even_mad_calc", "assert detect_anomalies([1, 2, 3, 10, 20], 4, 2.0) == [False, False, False, False, True]"),
        ("p_zero_multiplier", "assert detect_anomalies([5, 5, 5, 6], 2, 0.0) == [False, False, False, True]"),
        ("p_window_slides_evicting_old", "assert detect_anomalies([0, 0, 2, 4, 6, 8, 10], 2, 3.0) == [False, False, False, False, False, False, False]"),
        ("p_repeated_spikes_blocked", "assert detect_anomalies([1, 1, 1, 50, 50, 50, 2], 3, 2.0) == [False, False, False, True, True, True, False]"),
        ("p_floats_exact", "assert detect_anomalies([1.5, 2.5, 3.5, 10.0], 3, 2.0) == [False, False, False, True]"),
        ("p_short_input", "assert detect_anomalies([100, 200], 5, 1.0) == [False, False]"),
        ("p_nan_mid_warmup", "assert detect_anomalies([1, None, 1, None, 10], 2, 2.0) == [False, False, False, False, True]"),
        ("p_neg_baseline", "assert detect_anomalies([-10, -10, -10, -2], 3, 2.0) == [False, False, False, True]"),
    ),
    reference_solution="""
from collections import deque
import math

def _median(vals):
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return float(s[m]) if n % 2 == 1 else 0.5 * (s[m - 1] + s[m])

def detect_anomalies(values, window_size, multiplier):
    hist = deque(maxlen=window_size)
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(False)
            continue
        fv = float(v)
        if len(hist) < window_size:
            out.append(False)
            hist.append(fv)
            continue
        med = _median(hist)
        mad = _median([abs(x - med) for x in hist])
        if abs(fv - med) > multiplier * max(mad, 1.0):
            out.append(True)
        else:
            out.append(False)
            hist.append(fv)
    return out
""",
))

# 8. Discount-Coupon Shortest Latency Router
_reg(CodingProblem(
    problem_id="coupon_dijkstra",
    title="Discount-Coupon Shortest Latency Router",
    func_name="min_coupon_latency",
    spec="""Write `min_coupon_latency(num_nodes, edges, blocked, src, dst)`:
- Directed graph on `0 .. num_nodes - 1`; `blocked` nodes are offline (if `src` or `dst` is out-of-range or blocked, return `-1`; if `src == dst` and unblocked, return `0`).
- `edges`: `(u, v, latency, discountable)` (`latency >= 0` int; ignore negative latencies or edges touching blocked/out-of-range nodes).
- You may use AT MOST ONE coupon along the path on an edge with `discountable == True`, halving its cost to `latency // 2`.
- Return minimum total latency or `-1` if unreachable.""",
    staging_checks=(
        ("basic_coupon", "assert min_coupon_latency(3, [(0, 1, 9, True), (1, 2, 4, True)], [], 0, 2) == 8"),
        ("non_discountable_cant_use_coupon", "assert min_coupon_latency(2, [(0, 1, 10, False)], [], 0, 1) == 10"),
        ("blocked_src_even_if_same", "assert min_coupon_latency(3, [], [1], 1, 1) == -1"),
        ("same_src_dst_unblocked", "assert min_coupon_latency(3, [], [], 1, 1) == 0"),
        ("coupon_changes_optimal_route", "assert min_coupon_latency(3, [(0, 2, 10, False), (0, 1, 1, False), (1, 2, 14, True)], [], 0, 2) == 8"),
        ("blocked_intermediate", "assert min_coupon_latency(3, [(0, 1, 2, True), (1, 2, 2, True), (0, 2, 9, False)], [1], 0, 2) == 9"),
        ("unreachable", "assert min_coupon_latency(4, [(0, 1, 5, True)], [], 0, 3) == -1"),
        ("zero_latency_edge", "assert min_coupon_latency(2, [(0, 1, 0, True)], [], 0, 1) == 0"),
    ),
    prod_checks=(
        ("p_out_of_range_endpoints", "assert min_coupon_latency(3, [], [], -1, 2) == -1 and min_coupon_latency(3, [], [], 0, 3) == -1"),
        ("p_blocked_dst", "assert min_coupon_latency(3, [(0, 2, 4, True)], [2], 0, 2) == -1"),
        ("p_parallel_edges_diff_discountable", "assert min_coupon_latency(2, [(0, 1, 10, False), (0, 1, 13, True)], [], 0, 1) == 6"),
        ("p_neg_edge_ignored", "assert min_coupon_latency(2, [(0, 1, -5, True), (0, 1, 7, True)], [], 0, 1) == 3"),
        ("p_out_of_range_edge_ignored", "assert min_coupon_latency(3, [(0, 99, 1, True), (0, 2, 6, False)], [], 0, 2) == 6"),
        ("p_odd_floor_div", "assert min_coupon_latency(2, [(0, 1, 7, True)], [], 0, 1) == 3"),
        ("p_save_coupon_for_big_edge", "assert min_coupon_latency(4, [(0, 1, 5, True), (1, 2, 3, True), (2, 3, 21, True)], [], 0, 3) == 18"),
        ("p_cycle_graph", "assert min_coupon_latency(3, [(0, 1, 4, True), (1, 0, 1, True), (1, 2, 8, True)], [], 0, 2) == 8"),
        ("p_directed_not_undirected", "assert min_coupon_latency(2, [(1, 0, 2, True)], [], 0, 1) == -1"),
        ("p_zero_nodes", "assert min_coupon_latency(0, [], [], 0, 0) == -1"),
        ("p_multi_hop_zero", "assert min_coupon_latency(3, [(0, 1, 0, False), (1, 2, 1, True)], [], 0, 2) == 0"),
        ("p_two_routes_tie", "assert min_coupon_latency(4, [(0, 1, 10, True), (1, 3, 5, False), (0, 2, 6, False), (2, 3, 9, True)], [], 0, 3) == 10"),
    ),
    reference_solution="""
import heapq

def min_coupon_latency(num_nodes, edges, blocked, src, dst):
    blk = set(blocked)
    if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
        return -1
    if src in blk or dst in blk:
        return -1
    if src == dst:
        return 0
    adj = [[] for _ in range(num_nodes)]
    for u, v, lat, disc in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes and lat >= 0 and u not in blk and v not in blk:
            adj[u].append((v, int(lat), bool(disc)))
    INF = 10**18
    dist = [[INF, INF] for _ in range(num_nodes)]
    dist[src][0] = 0
    pq = [(0, src, 0)]
    while pq:
        d, u, used = heapq.heappop(pq)
        if d > dist[u][used]:
            continue
        if u == dst:
            return d
        for v, w, disc in adj[u]:
            nd = d + w
            if nd < dist[v][used]:
                dist[v][used] = nd
                heapq.heappush(pq, (nd, v, used))
            if used == 0 and disc:
                cd = d + (w // 2)
                if cd < dist[v][1]:
                    dist[v][1] = cd
                    heapq.heappush(pq, (cd, v, 1))
    ans = min(dist[dst])
    return -1 if ans >= INF else ans
""",
))


# ==============================================================================
# PART B: 8 LEGACY PRODUCTION INCIDENT BUG-HUNT MODULES (COMPLEX_PROBLEMS)
# ==============================================================================

# 9. Mid-Cycle SaaS Plan Proration & Credit-Note Engine
_reg(CodingProblem(
    problem_id="proration_invoice",
    title="Mid-Cycle SaaS Plan Proration & Credit-Note Engine",
    func_name="close_billing_cycle",
    spec="""Incident FIN-209: Enterprise finance flagged invoice discrepancies in `close_billing_cycle` when seats/tiers change mid-month, coupons apply on top of credit notes, or customers pause/downgrade to zero seats. Audit the legacy billing module below and fix all calculation and line-item counting bugs while preserving its signature and return tuple `(net_cash_due_cents, ending_credit_balance_cents, line_items_count)`.

```python
def close_billing_cycle(cycle_days, initial_credit_cents, coupon_pct, events):
    # events: (day_index_0_based, kind, val1, val2)
    #   ("PLAN", seats, daily_rate_cents_per_seat)  -> effective from day_index through next PLAN or cycle_days
    #   ("ADDON", flat_cents, discountable_bool)    -> one-time charge; non-discountable addons bypass coupon_pct
    #   ("CREDIT_GRANT", cents, expires_end_of_cycle_bool)
    # Finance ledger invariants:
    # - Events may arrive unsorted; valid days are [0, cycle_days). On the same day, the last PLAN in input order governs that day onward;
    #   contiguous active days sharing an identical positive (seats, rate) plan coalesce into ONE billed plan line item (0-seat or 0-rate spans produce no line item).
    # - `coupon_pct` (0..100 int) discounts `(seat_subtotal + discountable_addons)` with integer half-up rounding on the discount amount `floor((sub * pct + 50) / 100)`.
    # - Expiring cycle credits apply BEFORE persistent account credits; unused expiring credits vanish at cycle close, while unused persistent credits remain in `ending_credit_balance_cents`.
    events = sorted(events, key=lambda x: x[0])
    seats = 0
    rate = 0
    last_day = 0
    seat_subtotal = 0
    disc_addons = 0
    fixed_addons = 0
    persist_cred = initial_credit_cents
    expiring_cred = 0
    lines = 0

    for day, kind, v1, v2 in events:
        if day < 0 or day > cycle_days:
            continue
        if kind == "PLAN":
            if seats > 0:
                seat_subtotal += (day - last_day) * seats * rate
                lines += 1
            seats = v1
            rate = v2
            last_day = day
        elif kind == "ADDON":
            if v1 > 0:
                lines += 1
                if v2:
                    disc_addons += v1
                else:
                    fixed_addons += v1
        elif kind == "CREDIT_GRANT":
            if v1 > 0:
                if v2:
                    expiring_cred += v1
                else:
                    persist_cred += v1

    if seats > 0:
        seat_subtotal += (cycle_days - last_day) * seats * rate
        lines += 1

    discountable = seat_subtotal + disc_addons
    discounted = round(discountable * (100 - coupon_pct) / 100.0)
    gross = discounted + fixed_addons

    use_persist = min(persist_cred, gross)
    gross -= use_persist
    persist_cred -= use_persist

    use_exp = min(expiring_cred, gross)
    gross -= use_exp

    return gross, persist_cred, lines
```""",
    staging_checks=(
        ("s1_expiring_used_first", "assert close_billing_cycle(10, 500, 0, [(0, 'PLAN', 1, 100), (2, 'CREDIT_GRANT', 400, True)]) == (100, 0, 1)"),
        ("s2_expiring_preserves_persistent", "assert close_billing_cycle(10, 500, 0, [(0, 'PLAN', 1, 30), (1, 'CREDIT_GRANT', 400, True)]) == (0, 500, 1)"),
        ("s3_half_up_discount_rounding", "assert close_billing_cycle(1, 0, 50, [(0, 'PLAN', 1, 5)]) == (2, 0, 1)"),
        ("s4_day_equal_cycle_days_ignored", "assert close_billing_cycle(10, 0, 0, [(0, 'PLAN', 2, 10), (10, 'ADDON', 999, False)]) == (200, 0, 1)"),
        ("s5_same_day_plan_override_single_line", "assert close_billing_cycle(10, 0, 0, [(0, 'PLAN', 5, 100), (0, 'PLAN', 2, 50)]) == (1000, 0, 1)"),
        ("s6_downgrade_to_zero_then_resume", "assert close_billing_cycle(10, 0, 0, [(0, 'PLAN', 2, 100), (4, 'PLAN', 0, 100), (7, 'PLAN', 1, 100)]) == (1100, 0, 2)"),
        ("s7_nonrefundable_addon_bypasses_coupon", "assert close_billing_cycle(10, 0, 20, [(0, 'PLAN', 1, 100), (3, 'ADDON', 500, False)]) == (1300, 0, 2)"),
        ("s8_redundant_same_plan_coalesces_line", "assert close_billing_cycle(10, 0, 0, [(0, 'PLAN', 2, 50), (5, 'PLAN', 2, 50)]) == (1000, 0, 1)"),
    ),
    prod_checks=(
        ("p1_unsorted_same_day_plan_order", "assert close_billing_cycle(5, 0, 0, [(3, 'PLAN', 1, 10), (0, 'PLAN', 4, 10), (0, 'PLAN', 2, 10)]) == (80, 0, 2)"),
        ("p2_half_up_exact_midpoints", "assert close_billing_cycle(1, 0, 50, [(0, 'PLAN', 1, 5)]) == (2, 0, 1) and close_billing_cycle(1, 0, 10, [(0, 'PLAN', 1, 25)]) == (22, 0, 1)"),
        ("p3_expiring_vanishes_persistent_stays", "assert close_billing_cycle(5, 300, 0, [(0, 'PLAN', 1, 20), (0, 'CREDIT_GRANT', 250, True), (1, 'CREDIT_GRANT', 100, False)]) == (0, 400, 1)"),
        ("p4_plan_on_last_valid_day", "assert close_billing_cycle(5, 0, 0, [(4, 'PLAN', 3, 10)]) == (30, 0, 1)"),
        ("p5_plan_at_cycle_days_boundary_ignored", "assert close_billing_cycle(5, 0, 0, [(0, 'PLAN', 1, 10), (5, 'PLAN', 9, 99)]) == (50, 0, 1)"),
        ("p6_zero_rate_not_billed_line", "assert close_billing_cycle(5, 0, 0, [(0, 'PLAN', 3, 0)]) == (0, 0, 0)"),
        ("p7_negative_day_ignored", "assert close_billing_cycle(4, 100, 0, [(-1, 'PLAN', 5, 100), (1, 'PLAN', 1, 50)]) == (50, 0, 1)"),
        ("p8_discountable_addon_rounded_with_seats", "assert close_billing_cycle(2, 0, 50, [(0, 'PLAN', 1, 3), (1, 'ADDON', 3, True)]) == (4, 0, 2)"),
        ("p9_zero_cycle_days", "assert close_billing_cycle(0, 250, 20, [(0, 'PLAN', 2, 100)]) == (0, 250, 0)"),
        ("p10_addon_zero_ignored", "assert close_billing_cycle(3, 0, 0, [(0, 'ADDON', 0, True), (0, 'PLAN', 1, 10)]) == (30, 0, 1)"),
        ("p11_full_100_coupon_still_charges_fixed_addon", "assert close_billing_cycle(3, 50, 100, [(0, 'PLAN', 2, 100), (1, 'ADDON', 200, False)]) == (150, 0, 2)"),
        ("p12_same_plan_rejoined_after_gap_is_two_lines", "assert close_billing_cycle(6, 0, 0, [(0, 'PLAN', 2, 10), (2, 'PLAN', 0, 10), (4, 'PLAN', 2, 10)]) == (80, 0, 2)"),
    ),
    reference_solution="""
def close_billing_cycle(cycle_days, initial_credit_cents, coupon_pct, events):
    if cycle_days <= 0:
        return 0, initial_credit_cents, 0
    valid = [(idx, ev) for idx, ev in enumerate(events) if 0 <= ev[0] < cycle_days]
    valid.sort(key=lambda x: (x[1][0], x[0]))
    daily_plan = [None] * cycle_days
    cur_plan = None
    disc_addons = 0
    fixed_addons = 0
    persist_cred = initial_credit_cents
    expiring_cred = 0
    addon_lines = 0

    by_day = {}
    for _, ev in valid:
        by_day.setdefault(ev[0], []).append(ev)

    for d in range(cycle_days):
        for _, kind, v1, v2 in by_day.get(d, ()):
            if kind == "PLAN":
                cur_plan = (v1, v2) if (v1 > 0 and v2 > 0) else None
            elif kind == "ADDON" and v1 > 0:
                addon_lines += 1
                if v2:
                    disc_addons += v1
                else:
                    fixed_addons += v1
            elif kind == "CREDIT_GRANT" and v1 > 0:
                if v2:
                    expiring_cred += v1
                else:
                    persist_cred += v1
        daily_plan[d] = cur_plan

    seat_subtotal = 0
    plan_lines = 0
    i = 0
    while i < cycle_days:
        p = daily_plan[i]
        j = i + 1
        while j < cycle_days and daily_plan[j] == p:
            j += 1
        if p is not None:
            seat_subtotal += (j - i) * p[0] * p[1]
            plan_lines += 1
        i = j

    discountable = seat_subtotal + disc_addons
    discount_amt = (discountable * coupon_pct + 50) // 100
    gross = (discountable - discount_amt) + fixed_addons

    use_exp = min(expiring_cred, gross)
    gross -= use_exp
    use_per = min(persist_cred, gross)
    gross -= use_per
    persist_cred -= use_per

    return gross, persist_cred, plan_lines + addon_lines
""",
))

# 10. Distributed Lease Pool with Monotonic Fencing Tokens
_reg(CodingProblem(
    problem_id="lease_fencing",
    title="Distributed Lease Pool & Fencing-Token Guard",
    func_name="process_lease_ops",
    spec="""Incident DIST-118: Storage clients experienced split-brain writes and lost keys in `process_lease_ops`. Audit the lease manager below and fix all fencing, renewal, expiry, and tie-ordering bugs while keeping its return tuple `(results_list, committed_data_dict)`.

```python
def process_lease_ops(ttl, max_leases, ops):
    # Contract:
    # - Up to `max_leases >= 0` concurrent unexpired leases; `ttl > 0` gives half-open active interval [now, now + ttl).
    # - Process `ops` in ascending `now` timestamp order, preserving input order on timestamp ties!
    # - Monotonic `fencing_token` starts at 1 and increments ONLY when granting a brand-new lease (renewing an unexpired lease by the SAME owner extends expiry to `now + ttl` and keeps its existing `fencing_token`!).
    # - ("acquire", now, key, owner) -> appends fencing_token (int) or None.
    # - ("write", now, key, token, payload) -> succeeds iff `key` has an unexpired lease with `active_token == token`; persists `payload` in `committed_data_dict[key]` (data MUST survive subsequent lease release or expiry!) and appends True/False.
    # - ("release", now, key, owner, token) -> succeeds iff `key` has an unexpired lease held by `owner` AND `active_token == token`; frees the lease slot without deleting committed data and appends True/False.
    ops_sorted = sorted(ops, key=lambda x: x[1])
    leases = {}
    data = {}
    next_tok = 1
    out = []
    for op in ops_sorted:
        kind, now = op[0], op[1]
        for k in list(leases.keys()):
            if leases[k][1] < now:
                del leases[k]
        if kind == "acquire":
            _, _, k, owner = op
            if k in leases:
                if leases[k][0] == owner:
                    leases[k][1] = now + ttl
                    leases[k][2] = next_tok
                    next_tok += 1
                    out.append(leases[k][2])
                else:
                    out.append(None)
            elif len(leases) < max_leases and ttl > 0:
                leases[k] = [owner, now + ttl, next_tok]
                out.append(next_tok)
                next_tok += 1
            else:
                out.append(None)
        elif kind == "write":
            _, _, k, tok, payload = op
            if k in leases and leases[k][2] <= tok:
                data[k] = payload
                out.append(True)
            else:
                out.append(False)
        elif kind == "release":
            _, _, k, owner, tok = op
            if k in leases and leases[k][0] == owner:
                del leases[k]
                data.pop(k, None)
                out.append(True)
            else:
                out.append(False)
    return out, data
```""",
    staging_checks=(
        ("s1_renew_keeps_token_and_counter", "assert process_lease_ops(10, 2, [('acquire', 0, 'k1', 'a'), ('acquire', 4, 'k1', 'a'), ('acquire', 5, 'k2', 'b')]) == ([1, 1, 2], {})"),
        ("s2_exact_expiry_boundary", "assert process_lease_ops(5, 1, [('acquire', 0, 'db', 'a'), ('acquire', 5, 'db', 'b')]) == ([1, 2], {})"),
        ("s3_fencing_exact_equality", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('write', 1, 'k', 99, 'bad'), ('write', 2, 'k', 1, 'ok')]) == ([1, False, True], {'k': 'ok'})"),
        ("s4_release_keeps_committed_data", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('write', 1, 'k', 1, 'v1'), ('release', 2, 'k', 'a', 1)]) == ([1, True, True], {'k': 'v1'})"),
        ("s5_release_verifies_token", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('release', 1, 'k', 'a', 99), ('write', 2, 'k', 1, 'ok')]) == ([1, False, True], {'k': 'ok'})"),
        ("s6_unsorted_tie_preserves_input_order", "assert process_lease_ops(10, 1, [('acquire', 2, 'k2', 'b'), ('acquire', 2, 'k1', 'a'), ('acquire', 0, 'k0', 'x')]) == ([1, None, None], {})"),
        ("s7_tie_write_then_release", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('write', 5, 'k', 1, 'x'), ('release', 5, 'k', 'a', 1)]) == ([1, True, True], {'k': 'x'})"),
        ("s8_zero_max_leases", "assert process_lease_ops(10, 0, [('acquire', 0, 'k', 'a')]) == ([None], {})"),
    ),
    prod_checks=(
        ("p1_stale_token_rejected", "assert process_lease_ops(5, 2, [('acquire', 0, 'k', 'a'), ('acquire', 5, 'k', 'b'), ('write', 6, 'k', 1, 'stale'), ('write', 6, 'k', 2, 'fresh')]) == ([1, 2, False, True], {'k': 'fresh'})"),
        ("p2_tie_release_before_write", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('release', 5, 'k', 'a', 1), ('write', 5, 'k', 1, 'late')]) == ([1, True, False], {})"),
        ("p3_expiry_frees_slot_at_exact_now", "assert process_lease_ops(4, 1, [('acquire', 0, 'k1', 'a'), ('acquire', 4, 'k2', 'b')]) == ([1, 2], {})"),
        ("p4_data_survives_expiry", "assert process_lease_ops(3, 2, [('acquire', 0, 'k', 'a'), ('write', 1, 'k', 1, 'v1'), ('acquire', 5, 'k', 'b'), ('write', 6, 'k', 1, 'bad')]) == ([1, True, 2, False], {'k': 'v1'})"),
        ("p5_release_expired_fails", "assert process_lease_ops(3, 2, [('acquire', 0, 'k', 'a'), ('release', 3, 'k', 'a', 1)]) == ([1, False], {})"),
        ("p6_write_expired_fails", "assert process_lease_ops(3, 2, [('acquire', 0, 'k', 'a'), ('write', 3, 'k', 1, 'v')]) == ([1, False], {})"),
        ("p7_release_wrong_owner_fails", "assert process_lease_ops(10, 2, [('acquire', 0, 'k', 'a'), ('release', 2, 'k', 'b', 1)]) == ([1, False], {})"),
        ("p8_reacquire_after_release_increments_token", "assert process_lease_ops(10, 1, [('acquire', 0, 'k', 'a'), ('release', 2, 'k', 'a', 1), ('acquire', 2, 'k', 'a')]) == ([1, True, 2], {})"),
        ("p9_renewal_extends_past_original_expiry", "assert process_lease_ops(5, 1, [('acquire', 0, 'k', 'a'), ('acquire', 4, 'k', 'a'), ('write', 8, 'k', 1, 'ok')]) == ([1, 1, True], {'k': 'ok'})"),
        ("p10_contended_acquire_blocked_until_expiry", "assert process_lease_ops(5, 2, [('acquire', 0, 'k', 'a'), ('acquire', 4, 'k', 'b'), ('acquire', 5, 'k', 'b')]) == ([1, None, 2], {})"),
        ("p11_nonpositive_ttl_never_grants", "assert process_lease_ops(0, 5, [('acquire', 0, 'k', 'a')]) == ([None], {})"),
        ("p12_empty_ops", "assert process_lease_ops(10, 3, []) == ([], {})"),
    ),
    reference_solution="""
def process_lease_ops(ttl, max_leases, ops):
    indexed = sorted(enumerate(ops), key=lambda x: (x[1][1], x[0]))
    leases = {}
    data = {}
    next_tok = 1
    out = []
    for _, op in indexed:
        kind, now = op[0], op[1]
        for k in [k for k, v in leases.items() if v[1] <= now]:
            del leases[k]
        if kind == "acquire":
            _, _, k, owner = op
            if ttl <= 0:
                out.append(None)
            elif k in leases:
                if leases[k][0] == owner:
                    leases[k][1] = now + ttl
                    out.append(leases[k][2])
                else:
                    out.append(None)
            elif len(leases) < max_leases:
                leases[k] = [owner, now + ttl, next_tok]
                out.append(next_tok)
                next_tok += 1
            else:
                out.append(None)
        elif kind == "write":
            _, _, k, tok, payload = op
            if k in leases and leases[k][2] == tok:
                data[k] = payload
                out.append(True)
            else:
                out.append(False)
        elif kind == "release":
            _, _, k, owner, tok = op
            if k in leases and leases[k][0] == owner and leases[k][2] == tok:
                del leases[k]
                out.append(True)
            else:
                out.append(False)
    return out, data
""",
))

# 11. Reservation-Hold & Period-Rollover Quota Ledger
_reg(CodingProblem(
    problem_id="quota_rollover",
    title="Reservation-Hold & Period-Rollover Quota Ledger",
    func_name="settle_quota_ledger",
    spec="""Incident FIN-314: Teams hit phantom quota exhaustion and negative org usage in `settle_quota_ledger` when pre-authorization holds expire or settle across 100-second billing periods. Fix all rollover, hold-settlement, and refund bugs while preserving the return tuple `(decisions, final_team_available_dict)`.

```python
def settle_quota_ledger(org_cap, rollover_cap, team_grants, events):
    # Contract:
    # - Period p = timestamp // 100 (timestamp >= 0; sort ascending by timestamp, preserving input order on ties).
    # - Period rollover (advance step-by-step for each elapsed period!):
    #   * Any STILL-OPEN pre-auth hold created in an earlier period expires automatically at the period boundary BEFORE computing rollover (restoring its held amount back to that team's prior-period unused balance and removing it from prior-period org usage!).
    #   * Then each team starts the new period with `team_grants[t] + min(rollover_cap, prior_period_unused_balance[t])`, and `org_period_used` resets to 0.
    # - Events:
    #   * ("hold", ts, hold_id, team, amt): requires `team in team_grants`, `amt > 0`, `hold_id` not currently open, `team_avail >= amt`, and `org_used + amt <= org_cap`; reserves `amt` from both team & org and appends True (else False).
    #   * ("capture", ts, hold_id, actual_amt): if `hold_id` is open in the CURRENT period and `0 <= actual_amt <= held_amt`, finalizes spend at `actual_amt`, immediately releases unused `(held_amt - actual_amt)` back to team & org, closes `hold_id`, and appends True (if `actual_amt > held_amt`, `actual_amt < 0`, or `hold_id` not open, leave open hold untouched and append False!).
    #   * ("void", ts, hold_id): if `hold_id` is currently open, restores full `held_amt` to team & org, closes `hold_id`, appends True; else False.
    bal = dict(team_grants)
    cur_p = 0
    org_used = 0
    holds = {}  # hid -> (team, amt, period)
    out = []

    for ev in sorted(events, key=lambda x: x[1]):
        kind, ts = ev[0], ev[1]
        p = ts // 100
        if p > cur_p:
            for t in bal:
                bal[t] = team_grants[t] + min(rollover_cap, bal[t])
            org_used = 0
            cur_p = p
        if kind == "hold":
            _, _, hid, team, amt = ev
            if team in bal and amt > 0 and bal[team] >= amt and org_used + amt <= org_cap:
                bal[team] -= amt
                org_used += amt
                holds[hid] = (team, amt, cur_p)
                out.append(True)
            else:
                out.append(False)
        elif kind == "capture":
            _, _, hid, actual = ev
            if hid in holds and actual <= holds[hid][1]:
                team, held, _ = holds.pop(hid)
                diff = held - actual
                bal[team] += diff
                org_used -= diff
                out.append(True)
            else:
                out.append(False)
        elif kind == "void":
            _, _, hid = ev
            if hid in holds:
                team, held, _ = holds.pop(hid)
                bal[team] += held
                org_used -= held
                out.append(True)
            else:
                out.append(False)
    return out, bal
```""",
    staging_checks=(
        ("s1_unsettled_hold_expires_into_rollover", "assert settle_quota_ledger(200, 30, {'a': 50}, [('hold', 10, 'h1', 'a', 40), ('capture', 110, 'h1', 20)]) == ([True, False], {'a': 80})"),
        ("s2_multi_period_gap_compounds_rollover", "assert settle_quota_ledger(500, 40, {'a': 25}, [('hold', 300, 'h1', 'a', 10)]) == ([True], {'a': 55})"),
        ("s3_duplicate_open_hold_id_rejected", "assert settle_quota_ledger(200, 20, {'a': 50}, [('hold', 1, 'h1', 'a', 10), ('hold', 2, 'h1', 'a', 10)]) == ([True, False], {'a': 40})"),
        ("s4_over_capture_leaves_hold_open", "assert settle_quota_ledger(200, 20, {'a': 50}, [('hold', 1, 'h1', 'a', 20), ('capture', 2, 'h1', 25), ('capture', 3, 'h1', 15)]) == ([True, False, True], {'a': 35})"),
        ("s5_negative_capture_rejected_hold_intact", "assert settle_quota_ledger(200, 20, {'a': 50}, [('hold', 1, 'h1', 'a', 20), ('capture', 2, 'h1', -5), ('void', 3, 'h1')]) == ([True, False, True], {'a': 50})"),
        ("s6_zero_capture_valid_and_releases_all", "assert settle_quota_ledger(200, 20, {'a': 50}, [('hold', 1, 'h1', 'a', 20), ('capture', 2, 'h1', 0)]) == ([True, True], {'a': 50})"),
        ("s7_org_cap_freed_by_partial_capture", "assert settle_quota_ledger(50, 20, {'a': 50, 'b': 50}, [('hold', 1, 'h1', 'a', 50), ('capture', 2, 'h1', 30), ('hold', 3, 'h2', 'b', 20)]) == ([True, True, True], {'a': 20, 'b': 30})"),
        ("s8_empty_events_keeps_period_zero", "assert settle_quota_ledger(100, 10, {'a': 40}, []) == ([], {'a': 40})"),
    ),
    prod_checks=(
        ("p1_unsorted_tie_order_preserved", "assert settle_quota_ledger(100, 20, {'a': 30}, [('hold', 5, 'h2', 'a', 20), ('void', 5, 'h1'), ('hold', 2, 'h1', 'a', 20)]) == ([True, False, True], {'a': 30})"),
        ("p2_reusing_hold_id_after_close_allowed", "assert settle_quota_ledger(100, 20, {'a': 50}, [('hold', 1, 'h1', 'a', 20), ('void', 2, 'h1'), ('hold', 3, 'h1', 'a', 30)]) == ([True, True, True], {'a': 20})"),
        ("p3_expired_hold_id_reusable_next_period", "assert settle_quota_ledger(100, 10, {'a': 50}, [('hold', 10, 'h1', 'a', 40), ('hold', 110, 'h1', 'a', 60)]) == ([True, True], {'a': 0})"),
        ("p4_void_expired_hold_fails", "assert settle_quota_ledger(100, 10, {'a': 50}, [('hold', 10, 'h1', 'a', 40), ('void', 110, 'h1')]) == ([True, False], {'a': 60})"),
        ("p5_unknown_team_and_zero_amt_rejected", "assert settle_quota_ledger(100, 10, {'a': 50}, [('hold', 1, 'h1', 'ghost', 10), ('hold', 2, 'h2', 'a', 0)]) == ([False, False], {'a': 50})"),
        ("p6_captured_spend_reduces_rollover_base", "assert settle_quota_ledger(200, 50, {'a': 50}, [('hold', 10, 'h1', 'a', 35), ('capture', 20, 'h1', 30), ('void', 120, 'none')]) == ([True, True, False], {'a': 70})"),
        ("p7_two_period_skip_after_open_hold", "assert settle_quota_ledger(200, 35, {'a': 30}, [('hold', 10, 'h1', 'a', 25), ('void', 210, 'h1')]) == ([True, False], {'a': 65})"),
        ("p8_double_void_second_false", "assert settle_quota_ledger(100, 10, {'a': 40}, [('hold', 1, 'h', 'a', 10), ('void', 2, 'h'), ('void', 3, 'h')]) == ([True, True, False], {'a': 40})"),
        ("p9_org_cap_blocks_across_teams", "assert settle_quota_ledger(40, 10, {'a': 30, 'b': 30}, [('hold', 1, 'h1', 'a', 25), ('hold', 2, 'h2', 'b', 20)]) == ([True, False], {'a': 5, 'b': 30})"),
        ("p10_rollover_zero_cap", "assert settle_quota_ledger(100, 0, {'a': 40}, [('void', 150, 'x')]) == ([False], {'a': 40})"),
        ("p11_tie_void_before_hold_succeeds", "assert settle_quota_ledger(100, 10, {'a': 20}, [('hold', 1, 'h1', 'a', 20), ('void', 5, 'h1'), ('hold', 5, 'h2', 'a', 20)]) == ([True, True, True], {'a': 0})"),
        ("p12_capture_after_void_fails", "assert settle_quota_ledger(100, 10, {'a': 30}, [('hold', 1, 'h1', 'a', 15), ('void', 2, 'h1'), ('capture', 3, 'h1', 10)]) == ([True, True, False], {'a': 30})"),
    ),
    reference_solution="""
def settle_quota_ledger(org_cap, rollover_cap, team_grants, events):
    bal = dict(team_grants)
    if not events:
        return [], bal
    indexed = sorted(enumerate(events), key=lambda x: (x[1][1], x[0]))
    cur_p = 0
    org_used = 0
    holds = {}
    out = []

    for _, ev in indexed:
        kind, ts = ev[0], ev[1]
        p = ts // 100
        while cur_p < p:
            for hid in list(holds.keys()):
                team, held = holds.pop(hid)
                bal[team] += held
                org_used -= held
            for t in bal:
                bal[t] = team_grants[t] + min(rollover_cap, bal[t])
            org_used = 0
            cur_p += 1

        if kind == "hold":
            _, _, hid, team, amt = ev
            if team in bal and amt > 0 and hid not in holds and bal[team] >= amt and org_used + amt <= org_cap:
                bal[team] -= amt
                org_used += amt
                holds[hid] = (team, amt)
                out.append(True)
            else:
                out.append(False)
        elif kind == "capture":
            _, _, hid, actual = ev
            if hid in holds and 0 <= actual <= holds[hid][1]:
                team, held = holds.pop(hid)
                diff = held - actual
                bal[team] += diff
                org_used -= diff
                out.append(True)
            else:
                out.append(False)
        elif kind == "void":
            _, _, hid = ev
            if hid in holds:
                team, held = holds.pop(hid)
                bal[team] += held
                org_used -= held
                out.append(True)
            else:
                out.append(False)
    return out, bal
""",
))

# 12. Multi-AZ Cloud SLA Credit Waterfall
_reg(CodingProblem(
    problem_id="sla_waterfall",
    title="Multi-AZ Cloud Outage & SLA Credit Waterfall",
    func_name="compute_sla_credit",
    spec="""Incident FIN-402: Enterprise cloud customers were over- and under-credited by `compute_sla_credit` during multi-zone outages. Audit the SLA engine below and fix all interval-intersection, micro-blip filtering, deductible, and credit-offset bugs while preserving the return tuple `(total_qualified_downtime_mins, sla_tier_pct, net_payout_cents)`.

```python
def compute_sla_credit(month_mins, monthly_bill_cents, min_quorum_zones, goodwill_cents, zone_outages):
    # Contract:
    # - `zone_outages`: dict `zone_id -> list of (start_min, end_min)` raw outage windows within a month `[0, month_mins)`.
    #   Clip each raw window to `[0, month_mins)` and merge overlapping/contiguous windows WITHIN the same zone FIRST!
    # - Cluster-wide outage occurs at minute `m` iff AT LEAST `min_quorum_zones` distinct zones are simultaneously down at `m`.
    # - Micro-blip exclusion: after coalescing contiguous cluster-wide down intervals `[a, b)`, discard any contiguous cluster-wide incident strictly shorter than 5 minutes (`b - a < 5`)!
    # - Sum remaining qualified cluster outage durations into `total_qualified_downtime_mins`.
    # - Compare exact integer uptime inequalities against `month_mins` without float rounding:
    #   * Uptime >= 99.9% (`qualified * 1000 <= month_mins * 1`): `sla_tier_pct = 0`
    #   * 99.0% <= Uptime < 99.9% (`qualified * 100 <= month_mins * 1`): `sla_tier_pct = 10`
    #   * 95.0% <= Uptime < 99.0% (`qualified * 100 <= month_mins * 5`): `sla_tier_pct = 25`
    #   * Uptime < 95.0%: `sla_tier_pct = 100`
    # - Gross credit = `(monthly_bill_cents * sla_tier_pct) // 100`; `net_payout_cents = max(0, min(monthly_bill_cents - goodwill_cents, gross_credit - goodwill_cents))` (prior `goodwill_cents >= 0` offsets gross SLA credit AND total combined credits can never exceed `monthly_bill_cents`!).
    events = []
    for z, spans in zone_outages.items():
        for s, e in spans:
            if e - s >= 5:
                events.append((max(0, s), 1))
                events.append((min(month_mins, e), -1))
    events.sort()
    down = 0
    qual = 0
    prev = 0
    for t, delta in events:
        if down >= min_quorum_zones:
            qual += t - prev
        down += delta
        prev = t

    uptime = (month_mins - qual) / float(month_mins)
    if uptime >= 0.999:
        pct = 0
    elif uptime >= 0.99:
        pct = 10
    elif uptime >= 0.95:
        pct = 25
    else:
        pct = 100
    gross = (monthly_bill_cents * pct) // 100
    return qual, pct, max(0, gross - goodwill_cents)
```""",
    staging_checks=(
        ("s1_same_zone_overlap_does_not_fake_quorum", "assert compute_sla_credit(10000, 10000, 2, 0, {'z1': [(0, 20), (5, 25)], 'z2': []}) == (0, 0, 0)"),
        ("s2_split_spans_in_zone_merge_before_5min_filter", "assert compute_sla_credit(10000, 10000, 1, 0, {'z1': [(10, 13), (13, 16)]}) == (6, 0, 0)"),
        ("s3_cluster_overlap_under_5min_discarded", "assert compute_sla_credit(10000, 10000, 2, 0, {'z1': [(0, 10)], 'z2': [(6, 20)]}) == (0, 0, 0)"),
        ("s4_exact_999_boundary_is_zero_pct", "assert compute_sla_credit(10000, 50000, 1, 0, {'z1': [(0, 10)]}) == (10, 0, 0) and compute_sla_credit(10000, 50000, 1, 0, {'z1': [(0, 11)]}) == (11, 10, 5000)"),
        ("s5_goodwill_caps_total_refund_at_bill", "assert compute_sla_credit(1000, 10000, 1, 12000, {'z1': [(0, 200)]}) == (200, 100, 0)"),
        ("s6_adjacent_quorum_handoff_coalesces_cluster_incident", "assert compute_sla_credit(10000, 10000, 2, 0, {'z1': [(0, 3)], 'z2': [(0, 6)], 'z3': [(3, 6)]}) == (6, 0, 0)"),
        ("s7_out_of_bounds_clipped_before_duration_check", "assert compute_sla_credit(100, 10000, 1, 0, {'z1': [(-20, 4)]}) == (0, 0, 0)"),
        ("s8_tier_25_and_goodwill_offset", "assert compute_sla_credit(1000, 20000, 1, 1500, {'z1': [(0, 20)]}) == (20, 25, 3500)"),
    ),
    prod_checks=(
        ("p1_exact_99_and_95_boundaries", "assert compute_sla_credit(1000, 10000, 1, 0, {'z1': [(0, 10)]}) == (10, 10, 1000) and compute_sla_credit(1000, 10000, 1, 0, {'z1': [(0, 50)]}) == (50, 25, 2500) and compute_sla_credit(1000, 10000, 1, 0, {'z1': [(0, 51)]}) == (51, 100, 10000)"),
        ("p2_huge_scale_exact_integer_threshold", "assert compute_sla_credit(10**9, 100000, 1, 0, {'z1': [(0, 10**6 + 1)]}) == (1000001, 10, 10000)"),
        ("p3_three_way_duplicate_spans_same_zone", "assert compute_sla_credit(1000, 10000, 2, 0, {'z1': [(0, 30), (0, 30), (5, 25)], 'z2': [(0, 4)]}) == (0, 0, 0)"),
        ("p4_inverted_or_empty_spans_ignored", "assert compute_sla_credit(1000, 10000, 1, 0, {'z1': [(50, 10), (20, 20)]}) == (0, 0, 0)"),
        ("p5_clip_right_boundary_under_5m", "assert compute_sla_credit(100, 10000, 1, 0, {'z1': [(97, 150)]}) == (0, 0, 0)"),
        ("p6_clip_right_boundary_5m_kept", "assert compute_sla_credit(100, 10000, 1, 0, {'z1': [(95, 150)]}) == (5, 25, 2500)"),
        ("p7_multiple_separate_5m_incidents_sum", "assert compute_sla_credit(1000, 10000, 1, 0, {'z1': [(0, 6), (20, 26), (40, 44)]}) == (12, 25, 2500)"),
        ("p8_quorum_3_of_4", "assert compute_sla_credit(1000, 10000, 3, 0, {'a': [(0, 20)], 'b': [(0, 20)], 'c': [(5, 15)], 'd': [(10, 25)]}) == (15, 25, 2500)"),
        ("p9_goodwill_exceeds_partial_tier", "assert compute_sla_credit(1000, 10000, 1, 1500, {'z1': [(0, 10)]}) == (10, 10, 0)"),
        ("p10_zero_bill", "assert compute_sla_credit(1000, 0, 1, 0, {'z1': [(0, 500)]}) == (500, 100, 0)"),
        ("p11_empty_zones", "assert compute_sla_credit(1000, 10000, 2, 0, {}) == (0, 0, 0)"),
        ("p12_handoff_under_5m_dropped", "assert compute_sla_credit(1000, 10000, 2, 0, {'z1': [(0, 2)], 'z2': [(0, 4)], 'z3': [(2, 4)]}) == (0, 0, 0)"),
    ),
    reference_solution="""
def compute_sla_credit(month_mins, monthly_bill_cents, min_quorum_zones, goodwill_cents, zone_outages):
    events = []
    for spans in zone_outages.values():
        clipped = []
        for s, e in spans:
            cs = max(0, s)
            ce = min(month_mins, e)
            if cs < ce:
                clipped.append((cs, ce))
        if not clipped:
            continue
        clipped.sort()
        merged = [list(clipped[0])]
        for cs, ce in clipped[1:]:
            if cs <= merged[-1][1]:
                if ce > merged[-1][1]:
                    merged[-1][1] = ce
            else:
                merged.append([cs, ce])
        for cs, ce in merged:
            events.append((cs, 1))
            events.append((ce, -1))

    events.sort(key=lambda x: (x[0], -x[1]))
    down = 0
    cluster_spans = []
    cur_start = None
    for t, delta in events:
        prev_down = down
        down += delta
        if prev_down < min_quorum_zones and down >= min_quorum_zones:
            cur_start = t
        elif prev_down >= min_quorum_zones and down < min_quorum_zones:
            if cur_start is not None and t > cur_start:
                if cluster_spans and cluster_spans[-1][1] == cur_start:
                    cluster_spans[-1][1] = t
                else:
                    cluster_spans.append([cur_start, t])
            cur_start = None

    qual = sum(e - s for s, e in cluster_spans if (e - s) >= 5)
    if qual * 1000 <= month_mins:
        pct = 0
    elif qual * 100 <= month_mins:
        pct = 10
    elif qual * 100 <= month_mins * 5:
        pct = 25
    else:
        pct = 100

    gross = (monthly_bill_cents * pct) // 100
    max_allowed = max(0, monthly_bill_cents - goodwill_cents)
    net = max(0, min(max_allowed, gross - goodwill_cents))
    return qual, pct, net
""",
))

# 13. WAL Savepoint & Tombstone Compactor
_reg(CodingProblem(
    problem_id="wal_compactor",
    title="Write-Ahead Log Savepoint & Tombstone Compactor",
    func_name="replay_and_compact_wal",
    spec="""Incident DB-608: Crash recovery in `replay_and_compact_wal` resurrected deleted keys after compaction and corrupted state when rolling back nested savepoints. Fix the WAL engine below while preserving its return tuple `(visible_kv_dict, retained_wal_records_count)`.

```python
def replay_and_compact_wal(records, compact_up_to_lsn):
    # Contract:
    # - `records` have monotonically increasing `lsn` (int):
    #   * ("PUT", lsn, key, val)        -> sets key = val (val is never None)
    #   * ("DEL", lsn, key)             -> writes tombstone for key
    #   * ("SAVEPOINT", lsn, sp_name)   -> creates/overwrites savepoint `sp_name` capturing current uncommitted state
    #   * ("ROLLBACK_TO", lsn, sp_name) -> if `sp_name` exists in current uncommitted tx, restores state to `sp_name` AND invalidates any savepoints created AFTER `sp_name` (sp_name itself remains valid for repeated rollbacks!); if unknown `sp_name`, no-op!
    #   * ("COMMIT", lsn)               -> commits all surviving operations of the open batch at their original LSNs (rolled-back ops are discarded and NEVER enter committed log!), clearing all savepoints. Any trailing uncommitted ops after the last COMMIT are discarded!
    # - Compaction at `compact_up_to_lsn` over the COMMITTED log:
    #   * Partition committed ops into `historical` (`lsn <= compact_up_to_lsn`) and `recent` (`lsn > compact_up_to_lsn`).
    #   * Within `historical`, keep ONLY the latest committed op per key — AND drop historical `"DEL"` tombstones (since no older records below `compact_up_to_lsn` remain to shadow!).
    #   * Keep ALL committed ops in `recent` (`lsn > compact_up_to_lsn`), including `"DEL"` tombstones and overwritten puts!
    # - `visible_kv_dict` is the final live `{key: val}` state across all committed records (excluding deleted keys); `retained_wal_records_count` is `len(compacted_historical) + len(recent)`.
    committed = []
    pending = []
    savepoints = {}

    for rec in records:
        kind, lsn = rec[0], rec[1]
        if kind in ("PUT", "DEL"):
            pending.append(rec)
        elif kind == "SAVEPOINT":
            savepoints[rec[2]] = len(pending)
        elif kind == "ROLLBACK_TO":
            if rec[2] in savepoints:
                pending = pending[:savepoints.pop(rec[2])]
        elif kind == "COMMIT":
            committed.extend(pending)
            pending = []

    hist = {}
    recent = []
    for rec in committed:
        if rec[1] <= compact_up_to_lsn:
            if rec[0] == "PUT":
                hist[rec[2]] = rec
        else:
            recent.append(rec)

    state = {k: r[3] for k, r in hist.items()}
    for rec in recent:
        if rec[0] == "PUT":
            state[rec[2]] = rec[3]
        else:
            state.pop(rec[2], None)
    return state, len(hist) + len(recent)
```""",
    staging_checks=(
        ("s1_historical_del_must_erase_earlier_historical_put", "assert replay_and_compact_wal([('PUT', 1, 'k', 'v1'), ('DEL', 2, 'k'), ('COMMIT', 3)], 10) == ({}, 0)"),
        ("s2_savepoint_remains_reusable_after_rollback_to", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('SAVEPOINT', 2, 's1'), ('PUT', 3, 'b', 2), ('ROLLBACK_TO', 4, 's1'), ('PUT', 5, 'c', 3), ('ROLLBACK_TO', 6, 's1'), ('COMMIT', 7)], 0) == ({'a': 1}, 1)"),
        ("s3_rollback_invalidates_newer_savepoints", "assert replay_and_compact_wal([('SAVEPOINT', 1, 's1'), ('PUT', 2, 'a', 1), ('SAVEPOINT', 3, 's2'), ('ROLLBACK_TO', 4, 's1'), ('ROLLBACK_TO', 5, 's2'), ('COMMIT', 6)], 0) == ({}, 0)"),
        ("s4_commit_clears_prior_savepoints", "assert replay_and_compact_wal([('SAVEPOINT', 1, 's1'), ('PUT', 2, 'a', 1), ('COMMIT', 3), ('PUT', 4, 'b', 2), ('ROLLBACK_TO', 5, 's1'), ('COMMIT', 6)], 0) == ({'a': 1, 'b': 2}, 2)"),
        ("s5_recent_del_retained_in_wal_count", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('COMMIT', 2), ('DEL', 5, 'a'), ('COMMIT', 6)], 3) == ({}, 2)"),
        ("s6_uncommitted_tail_discarded", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('COMMIT', 2), ('PUT', 3, 'b', 2)], 10) == ({'a': 1}, 1)"),
        ("s7_overwritten_savepoint_name_moves_marker", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('SAVEPOINT', 2, 's'), ('PUT', 3, 'b', 2), ('SAVEPOINT', 4, 's'), ('PUT', 5, 'c', 3), ('ROLLBACK_TO', 6, 's'), ('COMMIT', 7)], 10) == ({'a': 1, 'b': 2}, 2)"),
        ("s8_recent_multiple_puts_kept_uncompacted", "assert replay_and_compact_wal([('PUT', 5, 'a', 1), ('PUT', 6, 'a', 2), ('COMMIT', 7)], 2) == ({'a': 2}, 2)"),
    ),
    prod_checks=(
        ("p1_put_del_put_all_historical_keeps_one", "assert replay_and_compact_wal([('PUT', 1, 'k', 'v1'), ('DEL', 2, 'k'), ('PUT', 3, 'k', 'v2'), ('COMMIT', 4)], 10) == ({'k': 'v2'}, 1)"),
        ("p2_del_at_exact_compact_boundary_dropped", "assert replay_and_compact_wal([('PUT', 1, 'k', 1), ('DEL', 5, 'k'), ('COMMIT', 6)], 5) == ({}, 0)"),
        ("p3_del_one_past_boundary_kept", "assert replay_and_compact_wal([('PUT', 1, 'k', 1), ('DEL', 6, 'k'), ('COMMIT', 7)], 5) == ({}, 2)"),
        ("p4_savepoint_overwritten_then_outer_rollback_drops_it", "assert replay_and_compact_wal([('SAVEPOINT', 1, 'outer'), ('PUT', 2, 'x', 1), ('SAVEPOINT', 3, 'inner'), ('PUT', 4, 'y', 2), ('SAVEPOINT', 5, 'inner'), ('ROLLBACK_TO', 6, 'outer'), ('ROLLBACK_TO', 7, 'inner'), ('COMMIT', 8)], 10) == ({}, 0)"),
        ("p5_rollback_erases_del_restoring_prior_put", "assert replay_and_compact_wal([('PUT', 1, 'k', 9), ('SAVEPOINT', 2, 'sp'), ('DEL', 3, 'k'), ('ROLLBACK_TO', 4, 'sp'), ('COMMIT', 5)], 10) == ({'k': 9}, 1)"),
        ("p6_unknown_savepoint_noop", "assert replay_and_compact_wal([('PUT', 1, 'k', 5), ('ROLLBACK_TO', 2, 'missing'), ('COMMIT', 3)], 10) == ({'k': 5}, 1)"),
        ("p7_multi_tx_historical_and_recent", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('PUT', 2, 'b', 2), ('COMMIT', 3), ('DEL', 4, 'a'), ('PUT', 8, 'b', 20), ('PUT', 9, 'c', 30), ('COMMIT', 10)], 5) == ({'b': 20, 'c': 30}, 3)"),
        ("p8_empty_records", "assert replay_and_compact_wal([], 100) == ({}, 0)"),
        ("p9_zero_value_preserved", "assert replay_and_compact_wal([('PUT', 1, 'z', 0), ('COMMIT', 2)], 5) == ({'z': 0}, 1)"),
        ("p10_del_nonexistent_historical_leaves_zero", "assert replay_and_compact_wal([('DEL', 1, 'ghost'), ('COMMIT', 2)], 5) == ({}, 0)"),
        ("p11_del_nonexistent_recent_retained", "assert replay_and_compact_wal([('DEL', 8, 'ghost'), ('COMMIT', 9)], 5) == ({}, 1)"),
        ("p12_uncommitted_savepoint_not_leaked", "assert replay_and_compact_wal([('PUT', 1, 'a', 1), ('SAVEPOINT', 2, 's1')], 10) == ({}, 0)"),
    ),
    reference_solution="""
def replay_and_compact_wal(records, compact_up_to_lsn):
    committed = []
    pending = []
    sp_stack = []

    for rec in records:
        kind = rec[0]
        if kind in ("PUT", "DEL"):
            pending.append(rec)
        elif kind == "SAVEPOINT":
            name = rec[2]
            sp_stack = [(n, pos) for (n, pos) in sp_stack if n != name]
            sp_stack.append((name, len(pending)))
        elif kind == "ROLLBACK_TO":
            name = rec[2]
            idx = next((i for i, (n, _) in enumerate(sp_stack) if n == name), None)
            if idx is not None:
                pos = sp_stack[idx][1]
                pending = pending[:pos]
                sp_stack = sp_stack[: idx + 1]
        elif kind == "COMMIT":
            committed.extend(pending)
            pending = []
            sp_stack = []

    hist_latest = {}
    recent = []
    for rec in committed:
        if rec[1] <= compact_up_to_lsn:
            hist_latest[rec[2]] = rec
        else:
            recent.append(rec)

    compacted_hist = {k: r for k, r in hist_latest.items() if r[0] == "PUT"}
    state = {k: r[3] for k, r in compacted_hist.items()}
    for rec in recent:
        if rec[0] == "PUT":
            state[rec[2]] = rec[3]
        else:
            state.pop(rec[2], None)
    return state, len(compacted_hist) + len(recent)
""",
))

# 14. Workflow DAG Retry & Join-Policy Executor
_reg(CodingProblem(
    problem_id="dag_retry_engine",
    title="Workflow Join-Policy & Backoff Retry Simulator",
    func_name="run_workflow_sim",
    spec="""Incident ORCH-771: Data pipelines hung or misfired downstream `"ONE_SUCCESS"` recovery branches in `run_workflow_sim` when upstream tasks exhausted retries or had zero parents. Fix the workflow simulator below while preserving its return tuple `(final_status_by_node, completion_time_by_node, total_attempts_run)`.

```python
def run_workflow_sim(spec_nodes, edges, attempt_outcomes):
    # Contract:
    # - `spec_nodes`: dict `node_id -> (duration_sec, max_retries, base_backoff_sec, join_policy)`
    #   where `duration_sec >= 1`, `max_retries >= 0` (total allowed attempts = 1 + max_retries), `join_policy` in `("ALL_SUCCESS", "ONE_SUCCESS")`.
    # - `edges`: `(parent_id, child_id)` valid DAG edges (ignore edges referencing unknown nodes; deduplicate edges).
    # - `attempt_outcomes`: dict `node_id -> list of bools` giving 1st, 2nd, ... attempt outcomes if executed (if list is shorter than attempts run, missing entries default to False).
    # - Root nodes (0 valid parents) ALWAYS become eligible to start at `t = 0` regardless of whether `join_policy` is `"ALL_SUCCESS"` or `"ONE_SUCCESS"`!
    # - Non-root node eligibility (barrier semantics: every node waits until ALL of its parents have reached a terminal state `{"SUCCESS", "FAILED", "SKIPPED"}`, at `ready_t = max(completion_time[p] for p in parents)`):
    #   * If `join_policy == "ALL_SUCCESS"`: runs at `ready_t` iff EVERY parent ended `"SUCCESS"`; otherwise immediately becomes `"SKIPPED"` at `completion_time = ready_t` (0 attempts run!).
    #   * If `join_policy == "ONE_SUCCESS"`: runs at `ready_t` iff AT LEAST ONE parent ended `"SUCCESS"`; otherwise immediately becomes `"SKIPPED"` at `completion_time = ready_t` (0 attempts run!).
    # - Execution & exponential backoff:
    #   * Attempt `k` (`k = 0` is initial try, `k = 1..max_retries` are retries): starts at `cur_t`, finishes at `cur_t + duration_sec`.
    #   * If `outcome` is True -> terminal `"SUCCESS"` at `cur_t + duration_sec`.
    #   * If `outcome` is False and `k < max_retries` -> sleeps `base_backoff_sec * (2 ** k)` after that attempt finishes, so next attempt `k + 1` starts at `(cur_t + duration_sec) + base_backoff_sec * (2 ** k)`.
    #   * If `outcome` is False and `k == max_retries` -> terminal `"FAILED"` at `cur_t + duration_sec`.
    parents = {u: [] for u in spec_nodes}
    for u, v in edges:
        if u in spec_nodes and v in spec_nodes:
            parents[v].append(u)

    status = {}
    done_t = {}
    attempts = 0

    while len(status) < len(spec_nodes):
        for u, (dur, retries, backoff, policy) in spec_nodes.items():
            if u in status:
                continue
            ps = parents[u]
            if any(p not in status for p in ps):
                continue
            t0 = max((done_t[p] for p in ps), default=0)
            if policy == "ALL_SUCCESS" and not all(status[p] == "SUCCESS" for p in ps):
                status[u] = "SKIPPED"
                done_t[u] = t0
                continue
            if policy == "ONE_SUCCESS" and not any(status[p] == "SUCCESS" for p in ps):
                status[u] = "SKIPPED"
                done_t[u] = t0
                continue
            outs = attempt_outcomes.get(u, [])
            t = t0
            ok = False
            for k in range(retries):
                attempts += 1
                t += dur
                if k < len(outs) and outs[k]:
                    ok = True
                    break
                t += backoff * (2 ** k)
            status[u] = "SUCCESS" if ok else "FAILED"
            done_t[u] = t
    return status, done_t, attempts
```""",
    staging_checks=(
        ("s1_root_with_one_success_policy_still_runs", "assert run_workflow_sim({'r': (5, 0, 2, 'ONE_SUCCESS')}, [], {'r': [True]}) == ({'r': 'SUCCESS'}, {'r': 5}, 1)"),
        ("s2_max_retries_zero_runs_once", "assert run_workflow_sim({'a': (4, 0, 10, 'ALL_SUCCESS')}, [], {'a': [False]}) == ({'a': 'FAILED'}, {'a': 4}, 1)"),
        ("s3_final_failed_attempt_does_not_add_trailing_backoff", "assert run_workflow_sim({'a': (5, 2, 10, 'ALL_SUCCESS')}, [], {'a': [False, False, False]}) == ({'a': 'FAILED'}, {'a': 45}, 3)"),
        ("s4_one_success_joins_mixed_parents", "assert run_workflow_sim({'p1': (3, 0, 1, 'ALL_SUCCESS'), 'p2': (8, 0, 1, 'ALL_SUCCESS'), 'j': (2, 0, 1, 'ONE_SUCCESS')}, [('p1', 'j'), ('p2', 'j')], {'p1': [False], 'p2': [True], 'j': [True]}) == ({'p1': 'FAILED', 'p2': 'SUCCESS', 'j': 'SUCCESS'}, {'p1': 3, 'p2': 8, 'j': 10}, 3)"),
        ("s5_skipped_cascades_cleanly", "assert run_workflow_sim({'a': (2, 0, 1, 'ALL_SUCCESS'), 'b': (3, 0, 1, 'ALL_SUCCESS'), 'c': (4, 0, 1, 'ONE_SUCCESS')}, [('a', 'b'), ('b', 'c')], {'a': [False], 'b': [True], 'c': [True]}) == ({'a': 'FAILED', 'b': 'SKIPPED', 'c': 'SKIPPED'}, {'a': 2, 'b': 2, 'c': 2}, 1)"),
        ("s6_retry_succeeds_on_last_allowed_try", "assert run_workflow_sim({'a': (3, 2, 4, 'ALL_SUCCESS')}, [], {'a': [False, False, True]}) == ({'a': 'SUCCESS'}, {'a': 21}, 3)"),
        ("s7_unknown_edge_ignored", "assert run_workflow_sim({'a': (2, 0, 1, 'ALL_SUCCESS')}, [('ghost', 'a')], {'a': [True]}) == ({'a': 'SUCCESS'}, {'a': 2}, 1)"),
        ("s8_missing_outcomes_default_false", "assert run_workflow_sim({'a': (2, 1, 5, 'ALL_SUCCESS')}, [], {}) == ({'a': 'FAILED'}, {'a': 9}, 2)"),
    ),
    prod_checks=(
        ("p1_zero_backoff_retries", "assert run_workflow_sim({'a': (3, 2, 0, 'ALL_SUCCESS')}, [], {'a': [False, True]}) == ({'a': 'SUCCESS'}, {'a': 6}, 2)"),
        ("p2_diamond_all_success_waits_slowest", "assert run_workflow_sim({'s': (1, 0, 0, 'ALL_SUCCESS'), 'l': (10, 0, 0, 'ALL_SUCCESS'), 'r': (3, 0, 0, 'ALL_SUCCESS'), 't': (2, 0, 0, 'ALL_SUCCESS')}, [('s', 'l'), ('s', 'r'), ('l', 't'), ('r', 't')], {'s': [True], 'l': [True], 'r': [True], 't': [True]}) == ({'s': 'SUCCESS', 'l': 'SUCCESS', 'r': 'SUCCESS', 't': 'SUCCESS'}, {'s': 1, 'l': 11, 'r': 4, 't': 13}, 4)"),
        ("p3_one_success_all_parents_failed_skips_at_max_parent_t", "assert run_workflow_sim({'p1': (4, 0, 1, 'ALL_SUCCESS'), 'p2': (9, 0, 1, 'ALL_SUCCESS'), 'j': (5, 0, 1, 'ONE_SUCCESS')}, [('p1', 'j'), ('p2', 'j')], {'p1': [False], 'p2': [False], 'j': [True]}) == ({'p1': 'FAILED', 'p2': 'FAILED', 'j': 'SKIPPED'}, {'p1': 4, 'p2': 9, 'j': 9}, 2)"),
        ("p4_duplicate_edges_safe", "assert run_workflow_sim({'a': (2, 0, 1, 'ALL_SUCCESS'), 'b': (3, 0, 1, 'ONE_SUCCESS')}, [('a', 'b'), ('a', 'b')], {'a': [True], 'b': [True]}) == ({'a': 'SUCCESS', 'b': 'SUCCESS'}, {'a': 2, 'b': 5}, 2)"),
        ("p5_empty_workflow", "assert run_workflow_sim({}, [('a', 'b')], {}) == ({}, {}, 0)"),
        ("p6_three_retries_backoff_progression", "assert run_workflow_sim({'x': (1, 3, 2, 'ALL_SUCCESS')}, [], {'x': [False, False, False, False]}) == ({'x': 'FAILED'}, {'x': 18}, 4)"),
        ("p7_parent_retry_delays_child_start", "assert run_workflow_sim({'a': (2, 1, 5, 'ALL_SUCCESS'), 'b': (3, 0, 1, 'ALL_SUCCESS')}, [('a', 'b')], {'a': [False, True], 'b': [True]}) == ({'a': 'SUCCESS', 'b': 'SUCCESS'}, {'a': 9, 'b': 12}, 3)"),
        ("p8_parent_failed_after_retries_sets_skip_time_without_extra_backoff", "assert run_workflow_sim({'a': (2, 1, 10, 'ALL_SUCCESS'), 'b': (5, 0, 1, 'ALL_SUCCESS')}, [('a', 'b')], {'a': [False, False], 'b': [True]}) == ({'a': 'FAILED', 'b': 'SKIPPED'}, {'a': 14, 'b': 14}, 2)"),
        ("p9_one_success_with_one_skipped_and_one_success", "assert run_workflow_sim({'r1': (2, 0, 1, 'ALL_SUCCESS'), 'm1': (2, 0, 1, 'ALL_SUCCESS'), 'r2': (7, 0, 1, 'ALL_SUCCESS'), 'end': (1, 0, 1, 'ONE_SUCCESS')}, [('r1', 'm1'), ('m1', 'end'), ('r2', 'end')], {'r1': [False], 'm1': [True], 'r2': [True], 'end': [True]}) == ({'r1': 'FAILED', 'm1': 'SKIPPED', 'r2': 'SUCCESS', 'end': 'SUCCESS'}, {'r1': 2, 'm1': 2, 'r2': 7, 'end': 8}, 3)"),
        ("p10_short_outcome_list_pads_false", "assert run_workflow_sim({'a': (2, 2, 3, 'ALL_SUCCESS')}, [], {'a': [False]}) == ({'a': 'FAILED'}, {'a': 15}, 3)"),
        ("p11_extra_outcomes_ignored_after_success", "assert run_workflow_sim({'a': (4, 2, 3, 'ALL_SUCCESS')}, [], {'a': [True, False, False]}) == ({'a': 'SUCCESS'}, {'a': 4}, 1)"),
        ("p12_multiple_roots_mixed_policies", "assert run_workflow_sim({'a': (3, 0, 1, 'ONE_SUCCESS'), 'b': (5, 0, 1, 'ALL_SUCCESS')}, [], {'a': [True], 'b': [True]}) == ({'a': 'SUCCESS', 'b': 'SUCCESS'}, {'a': 3, 'b': 5}, 2)"),
    ),
    reference_solution="""
def run_workflow_sim(spec_nodes, edges, attempt_outcomes):
    parents = {u: set() for u in spec_nodes}
    for u, v in edges:
        if u in spec_nodes and v in spec_nodes and u != v:
            parents[v].add(u)

    status = {}
    done_t = {}
    attempts = 0

    while len(status) < len(spec_nodes):
        progressed = False
        for u, (dur, retries, backoff, policy) in spec_nodes.items():
            if u in status:
                continue
            ps = parents[u]
            if any(p not in status for p in ps):
                continue
            progressed = True
            t0 = max((done_t[p] for p in ps), default=0)
            if ps:
                if policy == "ALL_SUCCESS" and not all(status[p] == "SUCCESS" for p in ps):
                    status[u] = "SKIPPED"
                    done_t[u] = t0
                    continue
                if policy == "ONE_SUCCESS" and not any(status[p] == "SUCCESS" for p in ps):
                    status[u] = "SKIPPED"
                    done_t[u] = t0
                    continue
            outs = attempt_outcomes.get(u, [])
            t = t0
            ok = False
            for k in range(retries + 1):
                attempts += 1
                t += dur
                if k < len(outs) and outs[k]:
                    ok = True
                    break
                if k < retries:
                    t += backoff * (2 ** k)
            status[u] = "SUCCESS" if ok else "FAILED"
            done_t[u] = t
        if not progressed:
            break
    return status, done_t, attempts
""",
))

# 15. Dual-Window Streaming Token Gateway Limiter
_reg(CodingProblem(
    problem_id="sliding_limiter",
    title="Dual-Window Streaming Token & Concurrency Gateway",
    func_name="evaluate_gateway_streams",
    spec="""Incident GW-819: Streaming LLM gateway (`evaluate_gateway_streams`) suffered concurrency leaks and retroactive rate-limit corruption when streams finished with fewer actual tokens than pre-reserved. Fix the gateway module below while preserving its return tuple `(admitted_ids_in_order, rejected_ids_in_order, active_streams_at_end)`.

```python
def evaluate_gateway_streams(burst_10s_cap, sustained_60s_cap, max_concurrent, events):
    # Contract:
    # - Sort events by ascending `ts` (int seconds); at the SAME `ts`, process ALL `"finish"` events BEFORE `"start"` events (preserving input order within each group).
    # - `("start", ts, req_id, reserved_tokens)` (`reserved_tokens > 0`):
    #   * Admitted iff:
    #     1) `active_concurrent < max_concurrent`,
    #     2) rolling 10s usage over `[ts - 9, ts]` inclusive + `reserved_tokens <= burst_10s_cap`,
    #     3) rolling 60s usage over `[ts - 59, ts]` inclusive + `reserved_tokens <= sustained_60s_cap`.
    #   * If admitted: record `req_id` in `admitted_ids_in_order`, mark active with `(start_ts=ts, reserved=reserved_tokens)`. While still active, it counts `reserved_tokens` anchored at `start_ts`.
    #   * If rejected: record `req_id` in `rejected_ids_in_order` (consumes 0 tokens and 0 concurrency).
    # - `("finish", ts, req_id, actual_tokens)`:
    #   * Ignored if `req_id` is not currently active.
    #   * Frees 1 concurrent slot immediately AND updates that request's token charge (still anchored at its original `start_ts`) to `max(0, actual_tokens)` (including if `actual_tokens > reserved_tokens`).
    active = {}
    ledger = []  # [start_ts, tokens]
    admitted = []
    rejected = []

    for ev in events:
        kind, ts = ev[0], ev[1]
        if kind == "start":
            _, _, rid, res_tok = ev
            u10 = sum(tok for t, tok in ledger if ts - t < 10)
            u60 = sum(tok for t, tok in ledger if ts - t <= 60)
            if len(active) < max_concurrent and u10 + res_tok <= burst_10s_cap and u60 + res_tok <= sustained_60s_cap:
                entry = [ts, res_tok]
                active[rid] = entry
                ledger.append(entry)
                admitted.append(rid)
            else:
                rejected.append(rid)
        elif kind == "finish":
            _, _, rid, act_tok = ev
            if rid in active:
                active[rid][1] = act_tok
    return admitted, rejected, len(active)
```""",
    staging_checks=(
        ("s1_finish_decrements_active_concurrency", "assert evaluate_gateway_streams(1000, 5000, 1, [('start', 0, 'r1', 100), ('finish', 2, 'r1', 80), ('start', 3, 'r2', 100)]) == (['r1', 'r2'], [], 1)"),
        ("s2_same_timestamp_finish_runs_before_start", "assert evaluate_gateway_streams(100, 500, 1, [('start', 0, 'r1', 100), ('start', 5, 'r2', 50), ('finish', 5, 'r1', 40)]) == (['r1', 'r2'], [], 1)"),
        ("s3_exact_60s_window_inclusive_59_exclusive_60", "assert evaluate_gateway_streams(1000, 100, 10, [('start', 0, 'r1', 100), ('start', 59, 'r2', 1), ('start', 60, 'r3', 100)]) == (['r1', 'r3'], ['r2'], 2)"),
        ("s4_exact_10s_window_inclusive_9_exclusive_10", "assert evaluate_gateway_streams(100, 1000, 10, [('start', 0, 'r1', 100), ('start', 9, 'r2', 1), ('start', 10, 'r3', 100)]) == (['r1', 'r3'], ['r2'], 2)"),
        ("s5_negative_actual_clamped_to_zero", "assert evaluate_gateway_streams(100, 100, 5, [('start', 0, 'r1', 80), ('finish', 1, 'r1', -50), ('start', 2, 'r2', 100)]) == (['r1', 'r2'], [], 1)"),
        ("s6_duplicate_or_unknown_finish_ignored", "assert evaluate_gateway_streams(100, 100, 5, [('start', 0, 'r1', 80), ('finish', 1, 'r1', 60), ('finish', 2, 'r1', 0), ('start', 3, 'r2', 50)]) == (['r1'], ['r2'], 0)"),
        ("s7_actual_overrun_above_reserved_recorded", "assert evaluate_gateway_streams(200, 200, 5, [('start', 0, 'r1', 50), ('finish', 1, 'r1', 180), ('start', 2, 'r2', 30)]) == (['r1'], ['r2'], 0)"),
        ("s8_zero_max_concurrent_rejects_all", "assert evaluate_gateway_streams(1000, 1000, 0, [('start', 0, 'r1', 10)]) == ([], ['r1'], 0)"),
    ),
    prod_checks=(
        ("p1_finish_unused_tokens_frees_window_budget", "assert evaluate_gateway_streams(100, 200, 5, [('start', 0, 'r1', 90), ('finish', 2, 'r1', 30), ('start', 3, 'r2', 70)]) == (['r1', 'r2'], [], 1)"),
        ("p2_charge_anchored_at_start_ts_not_finish_ts", "assert evaluate_gateway_streams(100, 500, 5, [('start', 0, 'r1', 90), ('finish', 9, 'r1', 90), ('start', 10, 'r2', 90)]) == (['r1', 'r2'], [], 1)"),
        ("p3_same_ts_multiple_starts_keep_input_order", "assert evaluate_gateway_streams(100, 500, 5, [('start', 5, 'b', 60), ('start', 5, 'a', 60)]) == (['b'], ['a'], 1)"),
        ("p4_finish_unknown_id_harmless", "assert evaluate_gateway_streams(100, 100, 2, [('finish', 0, 'ghost', 50), ('start', 1, 'r1', 100)]) == (['r1'], [], 1)"),
        ("p5_sustained_cap_blocks_after_multiple_10s_bursts", "assert evaluate_gateway_streams(100, 150, 10, [('start', 0, 'r1', 80), ('start', 15, 'r2', 80), ('start', 30, 'r3', 70)]) == (['r1', 'r3'], ['r2'], 2)"),
        ("p6_rejected_request_cannot_finish_later", "assert evaluate_gateway_streams(50, 50, 1, [('start', 0, 'r1', 50), ('start', 1, 'r2', 50), ('finish', 2, 'r2', 0), ('start', 3, 'r3', 10)]) == (['r1'], ['r2', 'r3'], 1)"),
        ("p7_empty_events", "assert evaluate_gateway_streams(100, 200, 3, []) == ([], [], 0)"),
        ("p8_exact_cap_equality_allowed", "assert evaluate_gateway_streams(100, 100, 2, [('start', 0, 'r1', 40), ('start', 5, 'r2', 60)]) == (['r1', 'r2'], [], 2)"),
        ("p9_same_ts_finish_frees_both_tokens_and_slot", "assert evaluate_gateway_streams(100, 100, 1, [('start', 0, 'r1', 100), ('start', 4, 'r2', 70), ('finish', 4, 'r1', 30)]) == (['r1', 'r2'], [], 1)"),
        ("p10_unsorted_timestamps_handled_cleanly", "assert evaluate_gateway_streams(100, 200, 2, [('start', 10, 'r2', 100), ('start', 0, 'r1', 100)]) == (['r1', 'r2'], [], 2)"),
        ("p11_overrun_expires_after_60s_from_start", "assert evaluate_gateway_streams(200, 200, 5, [('start', 0, 'r1', 50), ('finish', 20, 'r1', 200), ('start', 59, 'r2', 10), ('start', 60, 'r3', 200)]) == (['r1', 'r3'], ['r2'], 1)"),
        ("p12_three_concurrent_drained_to_zero", "assert evaluate_gateway_streams(500, 500, 3, [('start', 0, 'a', 10), ('start', 1, 'b', 10), ('start', 2, 'c', 10), ('finish', 3, 'a', 5), ('finish', 3, 'b', 5), ('finish', 3, 'c', 5)]) == (['a', 'b', 'c'], [], 0)"),
    ),
    reference_solution="""
def evaluate_gateway_streams(burst_10s_cap, sustained_60s_cap, max_concurrent, events):
    indexed = sorted(
        enumerate(events),
        key=lambda x: (x[1][1], 0 if x[1][0] == "finish" else 1, x[0]),
    )
    active = {}
    ledger = []
    admitted = []
    rejected = []

    for _, ev in indexed:
        kind, ts = ev[0], ev[1]
        if kind == "finish":
            _, _, rid, act_tok = ev
            entry = active.pop(rid, None)
            if entry is not None:
                entry[1] = max(0, int(act_tok))
        elif kind == "start":
            _, _, rid, res_tok = ev
            u10 = sum(tok for t, tok in ledger if ts - 9 <= t <= ts)
            u60 = sum(tok for t, tok in ledger if ts - 59 <= t <= ts)
            if (
                len(active) < max_concurrent
                and u10 + res_tok <= burst_10s_cap
                and u60 + res_tok <= sustained_60s_cap
            ):
                entry = [ts, int(res_tok)]
                active[rid] = entry
                ledger.append(entry)
                admitted.append(rid)
            else:
                rejected.append(rid)
    return admitted, rejected, len(active)
""",
))

# 16. Bitemporal Effective-Interval Price Book
_reg(CodingProblem(
    problem_id="bitemporal_book",
    title="Bitemporal Effective-Interval Price Override Book",
    func_name="query_bitemporal",
    spec="""Incident DATA-904: Historical billing audits in `query_bitemporal` returned dropped zero-price promotions, wrong precedence on batch-loaded same-transaction updates, and uncoalesced intervals. Fix the bitemporal query engine below while keeping its signature and return list of `(seg_start, seg_end, rate)` tuples.

```python
def query_bitemporal(assertions, as_of_tx, query_valid_range):
    # Contract:
    # - `assertions`: `(tx_time, valid_start, valid_end, rate)` over half-open `[valid_start, valid_end)` (ignore invalid tuples where `valid_start >= valid_end`).
    # - Visible iff `tx_time <= as_of_tx`. Higher `tx_time` overrides lower `tx_time`; on equal `tx_time`, the tuple appearing LATER in `assertions` wins!
    # - `rate is None` is a retraction (clears any prior rate on `[valid_start, valid_end)` so it produces a gap); numeric `0` or `0.0` is a VALID zero-price rate, NOT a retraction!
    # - Clip result to `query_valid_range = (q_start, q_end)` (if `q_start >= q_end`, return `[]`, and handle large sparse coordinates like `10**9` without O(range) memory!).
    # - Coalesce adjacent `[a, b)` and `[b, c)` segments when `rate_1 == rate_2`.
    qs, qe = query_valid_range
    if qs >= qe:
        return []
    vis = [a for a in assertions if a[0] <= as_of_tx and a[1] < a[2]]
    vis.sort(key=lambda x: x[0])
    pts = sorted({qs, qe} | {a[1] for a in vis} | {a[2] for a in vis})
    out = []
    for a, b in zip(pts, pts[1:]):
        if a < qs or b > qe:
            continue
        cur = None
        for tx, vs, ve, rate in vis:
            if vs <= a and b <= ve:
                cur = rate
        if cur:
            out.append((a, b, cur))
    return out
```""",
    staging_checks=(
        ("s1_override_splits_interval", "assert query_bitemporal([(1, 0, 100, 10), (2, 20, 50, 25)], 2, (0, 100)) == [(0, 20, 10), (20, 50, 25), (50, 100, 10)]"),
        ("s2_as_of_tx_hides_future", "assert query_bitemporal([(1, 0, 100, 10), (5, 20, 50, 25)], 3, (0, 100)) == [(0, 100, 10)]"),
        ("s3_retraction_creates_gap", "assert query_bitemporal([(1, 0, 100, 10), (2, 30, 70, None)], 2, (0, 100)) == [(0, 30, 10), (70, 100, 10)]"),
        ("s4_coalesce_adjacent_equal_rate", "assert query_bitemporal([(1, 0, 50, 10), (2, 25, 80, 10)], 2, (0, 100)) == [(0, 80, 10)]"),
        ("s5_same_tx_later_in_list_wins", "assert query_bitemporal([(5, 0, 50, 99), (1, 0, 50, 10), (5, 20, 40, 7)], 5, (0, 50)) == [(0, 20, 99), (20, 40, 7), (40, 50, 99)]"),
        ("s6_zero_rate_preserved_not_retracted", "assert query_bitemporal([(1, 0, 50, 10), (2, 10, 20, 0)], 2, (0, 50)) == [(0, 10, 10), (10, 20, 0), (20, 50, 10)]"),
        ("s7_query_clip_and_empty", "assert query_bitemporal([(1, 0, 100, 5)], 1, (20, 20)) == [] and query_bitemporal([(1, 0, 100, 5)], 1, (20, 40)) == [(20, 40, 5)]"),
        ("s8_retract_then_reassert_coalesces", "assert query_bitemporal([(1, 0, 100, 7), (2, 20, 60, None), (3, 20, 60, 7)], 3, (0, 100)) == [(0, 100, 7)]"),
    ),
    prod_checks=(
        ("p1_large_sparse_timestamps", "assert query_bitemporal([(1, 0, 10**9, 5), (2, 10**6, 2*10**6, None)], 2, (0, 10**9)) == [(0, 10**6, 5), (2*10**6, 10**9, 5)]"),
        ("p2_invalid_valid_interval_ignored", "assert query_bitemporal([(1, 0, 50, 5), (2, 30, 10, 99), (3, 20, 20, None)], 3, (0, 50)) == [(0, 50, 5)]"),
        ("p3_negative_coords", "assert query_bitemporal([(1, -100, 100, 3), (2, -20, 20, 8)], 2, (-50, 50)) == [(-50, -20, 3), (-20, 20, 8), (20, 50, 3)]"),
        ("p4_same_tx_retraction_after_assertion", "assert query_bitemporal([(1, 0, 40, 12), (1, 10, 30, None)], 1, (0, 40)) == [(0, 10, 12), (30, 40, 12)]"),
        ("p5_same_tx_assertion_after_retraction", "assert query_bitemporal([(1, 0, 40, None), (1, 10, 30, 0)], 1, (0, 40)) == [(10, 30, 0)]"),
        ("p6_adjacent_zeros_coalesce", "assert query_bitemporal([(1, 0, 20, 0), (2, 20, 40, 0.0)], 2, (0, 40)) == [(0, 40, 0)]"),
        ("p7_outside_query_window_ignored", "assert query_bitemporal([(1, 100, 200, 5)], 1, (0, 50)) == []"),
        ("p8_empty_assertions", "assert query_bitemporal([], 10, (0, 100)) == []"),
        ("p9_as_of_zero_filters_positive_tx", "assert query_bitemporal([(0, 0, 10, 4), (1, 0, 10, 9)], 0, (0, 10)) == [(0, 10, 4)]"),
        ("p10_inverted_query_range", "assert query_bitemporal([(1, 0, 100, 5)], 1, (50, 20)) == []"),
        ("p11_three_layer_override_coalesce", "assert query_bitemporal([(1, 0, 60, 10), (2, 20, 40, 20), (3, 10, 50, 10)], 3, (0, 60)) == [(0, 60, 10)]"),
        ("p12_partial_overlap_query_bounds", "assert query_bitemporal([(1, 10, 50, 8), (2, 30, 80, 8)], 2, (25, 60)) == [(25, 60, 8)]"),
    ),
    reference_solution="""
def query_bitemporal(assertions, as_of_tx, query_valid_range):
    qs, qe = query_valid_range
    if qs >= qe:
        return []
    vis = []
    for idx, (tx, vs, ve, rate) in enumerate(assertions):
        if tx <= as_of_tx and vs < ve:
            cs = max(qs, vs)
            ce = min(qe, ve)
            if cs < ce:
                vis.append((tx, idx, cs, ce, rate))
    if not vis:
        return []
    vis.sort(key=lambda x: (x[0], x[1]))
    coords = sorted({qs, qe} | {x[2] for x in vis} | {x[3] for x in vis})
    elem = [None] * (len(coords) - 1)
    idx_map = {c: i for i, c in enumerate(coords)}
    for _, _, cs, ce, rate in vis:
        for i in range(idx_map[cs], idx_map[ce]):
            elem[i] = rate
    out = []
    for i, r in enumerate(elem):
        if r is None:
            continue
        a, b = coords[i], coords[i + 1]
        if out and out[-1][1] == a and out[-1][2] == r:
            out[-1] = (out[-1][0], b, out[-1][2])
        else:
            out.append((a, b, r))
    return out
""",
))


def _worker_run_checks(code_str: str, checks: tuple[tuple[str, str], ...], conn: Any) -> None:
    """Isolated child-process runner so infinite loops or crashes never hang the harness."""
    ns: dict[str, Any] = {}
    try:
        exec(compile(code_str, "<candidate>", "exec"), ns, ns)
    except Exception as exc:
        conn.send({
            "compiled": False,
            "passed": 0,
            "total": len(checks),
            "all_passed": False,
            "failures": [f"Compile/Import Error: {type(exc).__name__}: {exc}"],
        })
        conn.close()
        return

    passed = 0
    failures: list[str] = []
    for label, snippet in checks:
        try:
            exec(compile(snippet, f"<check:{label}>", "exec"), ns, ns)
            passed += 1
        except Exception:
            tb_line = traceback.format_exc().strip().splitlines()[-1]
            failures.append(f"[{label}] {snippet} -> {tb_line}")

    conn.send({
        "compiled": True,
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "failures": failures,
    })
    conn.close()


def evaluate_code(
    code_str: str,
    problem: CodingProblem,
    suite: str = "prod",
    timeout_sec: float = 2.5,
) -> dict[str, Any]:
    """Execute candidate Python code against `staging` or `prod` checks in an isolated process."""
    checks = problem.staging_checks if suite == "staging" else problem.prod_checks
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker_run_checks, args=(code_str, checks, child_conn))
    proc.start()
    child_conn.close()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(0.5)
        parent_conn.close()
        return {
            "compiled": True,
            "passed": 0,
            "total": len(checks),
            "all_passed": False,
            "failures": [f"TimeoutError: exceeded {timeout_sec}s limit"],
        }
    if parent_conn.poll():
        res = parent_conn.recv()
        parent_conn.close()
        return res
    parent_conn.close()
    return {
        "compiled": False,
        "passed": 0,
        "total": len(checks),
        "all_passed": False,
        "failures": [f"Process crashed with exitcode={proc.exitcode}"],
    }


def validate_task_suite() -> dict[str, Any]:
    """Prove every reference solution passes 100% of staging (8) & prod (12) checks,
    trivial stubs fail 100%, and every legacy starter block fails before repair."""
    assert len(PROBLEMS) == 16, f"Expected 16 problems, got {len(PROBLEMS)}"
    buggy_modules_verified = 0
    for pid, prob in PROBLEMS.items():
        assert len(prob.staging_checks) == 8, f"{pid} must have 8 staging checks"
        assert len(prob.prod_checks) == 12, f"{pid} must have 12 prod checks"
        s_res = evaluate_code(prob.reference_solution, prob, suite="staging")
        assert s_res["all_passed"], f"{pid} staging failed on reference: {s_res['failures']}"
        p_res = evaluate_code(prob.reference_solution, prob, suite="prod")
        assert p_res["all_passed"], f"{pid} prod failed on reference: {p_res['failures']}"
        stub = f"def {prob.func_name}(*args, **kwargs):\n    return None\n"
        b_res = evaluate_code(stub, prob, suite="prod")
        assert not b_res["all_passed"], f"{pid} prod trivially passed on None stub!"

        fences = re.findall(r"```python\s*(.*?)```", prob.spec, re.S)
        if fences:
            buggy_src = fences[0]
            bs_stage = evaluate_code(buggy_src, prob, suite="staging")
            bs_prod = evaluate_code(buggy_src, prob, suite="prod")
            assert not bs_stage["all_passed"], f"{pid} starter code unexpectedly passed staging!"
            assert not bs_prod["all_passed"], f"{pid} starter code unexpectedly passed prod!"
            buggy_modules_verified += 1

    assert buggy_modules_verified == 8, f"Expected 8 legacy bug-hunt modules, got {buggy_modules_verified}"
    return {
        "num_problems": len(PROBLEMS),
        "staging_assertions": 128,
        "prod_assertions": 192,
        "legacy_bughunt_modules": buggy_modules_verified,
        "status": "ALL_16_VALID",
    }
