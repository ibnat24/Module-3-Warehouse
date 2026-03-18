"""
MSE 433 — Module 3 conveyor strategy simulation.

This script evaluates heuristics across three decision layers:
1) conveyor assignment, 2) tote sequencing, 3) within-tote item ordering.

Default behavior evaluates the full 3 x 3 x 4 grid (36 strategy combinations)
and ranks plans by makespan, then total completion time, then average completion time.

Simulation assumptions mirror the IDEAS Clinic setup:
- 4 conveyor stations in a loop
- item release rate is supply-constrained to one item every 5 seconds
- travel time between stations is 6 seconds (24 seconds full loop)
- only the active order at the front of each station queue can capture items
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Callable, Any
import heapq
import os
import csv


# ==========================
# TUNE THESE
# ==========================
TOP_N = 6
NUM_CONVEYORS = 4
TRAVEL_TIME = 6
ITEM_PLACE_TIME = 2
LOADING_BELT_TIME = 5
# Supply-constrained model: one item enters the system every 5 seconds.
RELEASE_INTERVAL = LOADING_BELT_TIME

# If you observed tote swapping takes time in real life, set this > 0
TOTE_CHANGEOVER_TIME = 0

MAX_PER_SHAPE_AVAILABLE = 8

SHAPE_NAMES = ["circle", "pentagon", "trapezoid", "triangle", "star", "moon", "heart", "cross"]
SIM_COLS = ["cirle", "pentagon", "trapezoid", "triangle", "star", "moon", "heart", "cross"]
NUM_SHAPES = 8

SHAPE_NAME_TO_COL = {
    "circle": "cirle",
    "pentagon": "pentagon",
    "trapezoid": "trapezoid",
    "triangle": "triangle",
    "star": "star",
    "moon": "moon",
    "heart": "heart",
    "cross": "cross",
}

# ==========================
# DEFAULT TEST MODE
# ==========================
# If True, tote and within-tote rules are fixed and only conveyor assignment is compared.
# If False, the full 3 x 3 x 4 heuristic grid is evaluated (36 combinations).
COMPARE_CONVEYOR_ONLY = False
FIXED_TOTE_RULE = "ID_ASC"
FIXED_WITHIN_RULE = "BPF"


# ==========================
# Data structures
# ==========================
@dataclass
class Order:
    id: int
    types: List[int]
    qtys: List[int]
    totes: List[int]


# ==========================
# CSV loading
# ==========================
def _read_rows(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            rows.append(parts)
    return rows


def load_orders_first_n(types_path: str, qty_path: str, totes_path: str, n: int) -> List[Order]:
    t = _read_rows(types_path)
    q = _read_rows(qty_path)
    r = _read_rows(totes_path)

    total = min(len(t), len(q), len(r), n)
    orders: List[Order] = []

    for i in range(total):
        types = [int(float(x)) for x in t[i] if x != ""]
        qtys = [int(float(x)) for x in q[i] if x != ""]
        totes = [int(float(x)) for x in r[i] if x != ""]

        m = min(len(types), len(qtys), len(totes))
        orders.append(Order(id=i, types=types[:m], qtys=qtys[:m], totes=totes[:m]))

    return orders


# ==========================
# Basic helpers
# ==========================
def order_demand_vector(o: Order) -> List[int]:
    dem = [0] * NUM_SHAPES
    for s, q in zip(o.types, o.qtys):
        if 0 <= s < NUM_SHAPES and q > 0:
            dem[s] += q
    return dem


def total_items_in_order(o: Order) -> int:
    return sum(order_demand_vector(o))


def unique_totes_in_order(o: Order) -> int:
    return len(set(o.totes))


def distinct_shapes_in_order(o: Order) -> int:
    return len(set(s for s, q in zip(o.types, o.qtys) if q > 0))


def fragmentation_score(o: Order) -> int:
    """
    Rough measure of operational complexity.
    Larger if the order spans many tote-shape combinations.
    """
    score = 0
    for s, q, t in zip(o.types, o.qtys, o.totes):
        if q > 0:
            score += 1
    return score


def build_tote_contents(orders: List[Order]) -> Dict[int, List[int]]:
    tote_contents = defaultdict(lambda: [0] * NUM_SHAPES)
    for o in orders:
        for s, q, t in zip(o.types, o.qtys, o.totes):
            if 0 <= s < NUM_SHAPES and q > 0:
                tote_contents[t][s] += q
    return dict(tote_contents)


# ==========================
# Tote sequencing heuristics
# ==========================
def totes_id_asc(all_totes: List[int], **_) -> List[int]:
    return sorted(all_totes)


def totes_mitf(all_totes: List[int], tote_contents: Dict[int, List[int]], **_) -> List[int]:
    return sorted(all_totes, key=lambda t: -sum(tote_contents[t]))


def totes_eocf(all_totes: List[int], tote_to_orders: Dict[int, List[int]], order_num_totes: Dict[int, int], **_) -> List[int]:
    def key(t: int) -> int:
        oids = tote_to_orders.get(t, [])
        return min(order_num_totes[oid] for oid in oids) if oids else 10**9
    return sorted(all_totes, key=key)


def get_tote_rule(name: str) -> Tuple[str, Callable]:
    mapping = {
        "ID_ASC": ("ID_ASC", totes_id_asc),
        "MITF": ("MITF", totes_mitf),
        "EOCF": ("EOCF", totes_eocf),
    }
    if name not in mapping:
        raise ValueError(f"Unknown tote rule: {name}")
    return mapping[name]


# ==========================
# Conveyor assignment heuristics
# ==========================
def assign_baseline_cycle(orders: List[Order]) -> Dict[int, int]:
    """
    Simple FCFS-style cycle assignment:
    order 0 -> conv 1
    order 1 -> conv 2
    ...
    """
    return {o.id: (o.id % NUM_CONVEYORS) + 1 for o in orders}


def estimate_order_workload(o: Order, wt_totes: float = 0.0) -> float:
    """
        Workload score used for LPT-based conveyor assignment.

        LPT_BALANCE:          score = total_items
        LPT_BALANCE_WT (2.0): score = total_items + 2 * unique_totes
    """
    items = total_items_in_order(o)
    totes = unique_totes_in_order(o)

    if wt_totes <= 0:
        return float(items)

    return float(items + wt_totes * totes)


def assign_lpt_balance(orders: List[Order], wt_totes: float = 0.0) -> Dict[int, int]:
    """
    Greedy load balancing:
      1. Score each order by estimated workload
      2. Assign biggest orders first to the currently lightest conveyor
    """
    scores: Dict[int, float] = {}
    for o in orders:
        scores[o.id] = estimate_order_workload(o, wt_totes=wt_totes)

    sorted_orders = sorted(orders, key=lambda o: -scores[o.id])

    load = {c: 0.0 for c in range(1, NUM_CONVEYORS + 1)}
    assign: Dict[int, int] = {}

    for o in sorted_orders:
        c_best = min(load.keys(), key=lambda c: load[c])
        assign[o.id] = c_best
        load[c_best] += scores[o.id]

    return assign


# ==========================
# Within-tote ordering heuristics
# ==========================
def shapes_naive(shape_qty: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return sorted(shape_qty, key=lambda x: x[0])


def shapes_bpf(shape_qty: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return sorted(shape_qty, key=lambda x: -x[1])


def shapes_flf(shape_qty: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return sorted(shape_qty, key=lambda x: -x[2])


def shapes_wflf(shape_qty: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    return sorted(shape_qty, key=lambda x: -(x[2] * x[1]))


def get_within_rule(name: str) -> Tuple[str, Callable]:
    mapping = {
        "NAIVE": ("NAIVE", shapes_naive),
        "BPF": ("BPF", shapes_bpf),
        "FLF": ("FLF", shapes_flf),
        "WFLF": ("WFLF", shapes_wflf),
    }
    if name not in mapping:
        raise ValueError(f"Unknown within-tote rule: {name}")
    return mapping[name]


# ==========================
# Build loading sequence
# ==========================
def build_loading_sequence(
    tote_sequence: List[int],
    tote_contents: Dict[int, List[int]],
    conv_assignment: Dict[int, int],
    orders: List[Order],
    within_tote_rule: Callable[[List[Tuple[int, int, int]]], List[Tuple[int, int, int]]],
) -> List[Dict[str, Any]]:
    """
    Builds the physical item release order.
    Each item is released one at a time, every RELEASE_INTERVAL seconds.
    """

    orders_need_shape = {s: set() for s in range(NUM_SHAPES)}
    for o in orders:
        dem = order_demand_vector(o)
        for s in range(NUM_SHAPES):
            if dem[s] > 0:
                orders_need_shape[s].add(o.id)

    seq: List[Dict[str, Any]] = []
    t_ptr = 0
    step = 1

    for tote_id in tote_sequence:
        cont = tote_contents.get(tote_id, [0] * NUM_SHAPES)

        shape_qty_lane: List[Tuple[int, int, int]] = []
        for s in range(NUM_SHAPES):
            q = cont[s]
            if q <= 0:
                continue

            lanes = [conv_assignment[oid] for oid in orders_need_shape[s]]
            approx_lane = max(lanes) if lanes else 1
            shape_qty_lane.append((s, q, approx_lane))

        if not shape_qty_lane:
            continue

        ordered_blocks = within_tote_rule(shape_qty_lane)

        for s, q, _lane in ordered_blocks:
            for _ in range(q):
                seq.append({
                    "step": step,
                    "time_release": t_ptr,
                    "tote_id": tote_id,
                    "shape_num": s,
                    "shape_name": SHAPE_NAMES[s],
                })
                step += 1
                t_ptr += RELEASE_INTERVAL

        t_ptr += TOTE_CHANGEOVER_TIME

    return seq


def write_loading_plan_csv(sequence: List[Dict[str, Any]], path: str) -> None:
    fieldnames = ["step", "time_release", "tote_id", "shape_num", "shape_name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sequence:
            w.writerow(row)


# ==========================
# Simulation
# ==========================
def simulate(
    orders: List[Order],
    conv_assignment: Dict[int, int],
    loading_sequence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Event-driven conveyor simulator.

    Logic:
    - items are injected at conveyor 1 according to the loading sequence
    - each move to next conveyor takes TRAVEL_TIME
    - each conveyor has a FIFO order queue
    - only the active order (front of queue) can capture
    - if it doesn't need the item, the item keeps circulating
    """

    remaining = {o.id: order_demand_vector(o)[:] for o in orders}
    remaining_total = {oid: sum(vec) for oid, vec in remaining.items()}
    total_needed = sum(remaining_total.values())

    queues = {c: deque() for c in range(1, NUM_CONVEYORS + 1)}
    for o in sorted(orders, key=lambda x: x.id):
        queues[conv_assignment[o.id]].append(o.id)

    def active_order(c: int) -> Optional[int]:
        while queues[c]:
            oid = queues[c][0]
            if remaining_total[oid] > 0:
                return oid
            queues[c].popleft()
        return None

    completion_time: Dict[int, Optional[int]] = {o.id: None for o in orders}

    events: List[Tuple[int, int, int]] = []
    for item in loading_sequence:
        heapq.heappush(events, (int(item["time_release"]), 1, int(item["shape_num"])))

    captured = 0

    while events and captured < total_needed:
        t, c, s = heapq.heappop(events)

        oid = active_order(c)
        if oid is not None and remaining[oid][s] > 0:
            remaining[oid][s] -= 1
            remaining_total[oid] -= 1
            captured += 1

            if remaining_total[oid] == 0 and completion_time[oid] is None:
                completion_time[oid] = t
        else:
            c_next = (c % NUM_CONVEYORS) + 1
            heapq.heappush(events, (t + TRAVEL_TIME, c_next, s))

    comp = [ct for ct in completion_time.values() if ct is not None]
    makespan = max(comp) if comp else 0
    total_ct = sum(comp) if comp else 0
    avg_ct = total_ct / len(comp) if comp else 0

    return {
        "makespan": makespan,
        "total_completion": total_ct,
        "avg_completion": avg_ct,
        "completion_time": {k: (v if v is not None else -1) for k, v in completion_time.items()},
    }


# ==========================
# Output writers
# ==========================
def write_sim_input(orders: List[Order], conv_assignment: Dict[int, int], path: str) -> None:
    header = "conv_num," + ",".join(SIM_COLS)
    lines = [header]

    for o in sorted(orders, key=lambda x: x.id):
        dem = order_demand_vector(o)
        conv_num = conv_assignment[o.id]
        row = [str(conv_num)] + [str(dem[i]) for i in range(NUM_SHAPES)]
        lines.append(",".join(row))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_strategy_results_csv(results: List[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "rank",
        "tote_rule",
        "conv_rule",
        "within_rule",
        "makespan",
        "total_completion",
        "avg_completion",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow({k: row[k] for k in fieldnames})


# ==========================
# Strategy search
# ==========================
def run_strategy_search(
    orders: List[Order]
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:

    tote_contents = build_tote_contents(orders)
    all_totes = sorted(tote_contents.keys())

    tote_to_orders = defaultdict(list)
    order_num_totes: Dict[int, int] = {}
    for o in orders:
        order_num_totes[o.id] = len(set(o.totes))
        for t in set(o.totes):
            tote_to_orders[t].append(o.id)

    if COMPARE_CONVEYOR_ONLY:
        tote_rules = [get_tote_rule(FIXED_TOTE_RULE)]
        within_rules = [get_within_rule(FIXED_WITHIN_RULE)]
    else:
        tote_rules = [
            ("ID_ASC", totes_id_asc),
            ("MITF", totes_mitf),
            ("EOCF", totes_eocf),
        ]
        within_rules = [
            ("NAIVE", shapes_naive),
            ("BPF", shapes_bpf),
            ("FLF", shapes_flf),
            ("WFLF", shapes_wflf),
        ]

    conv_rules = [
        ("BASELINE_CYCLE", lambda os_: assign_baseline_cycle(os_)),
        ("LPT_BALANCE", lambda os_: assign_lpt_balance(os_, wt_totes=0.0)),
        ("LPT_BALANCE_WT", lambda os_: assign_lpt_balance(os_, wt_totes=2.0)),
    ]

    # Baseline plan = fixed tote + baseline conveyor + fixed within
    baseline_tote_name, baseline_tote_fn = tote_rules[0]
    baseline_within_name, baseline_within_fn = within_rules[0]

    baseline_assign = assign_baseline_cycle(orders)
    baseline_totes = baseline_tote_fn(
        all_totes=all_totes,
        tote_contents=tote_contents,
        tote_to_orders=tote_to_orders,
        order_num_totes=order_num_totes,
    )
    baseline_loading = build_loading_sequence(
        baseline_totes,
        tote_contents,
        baseline_assign,
        orders,
        baseline_within_fn,
    )
    baseline_res = simulate(orders, baseline_assign, baseline_loading)

    baseline_plan = {
        "tote_rule": baseline_tote_name,
        "conv_rule": "BASELINE_CYCLE",
        "within_rule": baseline_within_name,
        "tote_sequence": baseline_totes,
        "conv_assignment": baseline_assign,
        "tote_contents": tote_contents,
        "loading_sequence": baseline_loading,
        "baseline_res": baseline_res,
    }

    all_results: List[Dict[str, Any]] = []

    for tote_name, tote_fn in tote_rules:
        tote_seq = tote_fn(
            all_totes=all_totes,
            tote_contents=tote_contents,
            tote_to_orders=tote_to_orders,
            order_num_totes=order_num_totes,
        )

        for conv_name, conv_fn in conv_rules:
            conv_assign = conv_fn(orders)

            for within_name, within_fn in within_rules:
                loading_seq = build_loading_sequence(
                    tote_seq,
                    tote_contents,
                    conv_assign,
                    orders,
                    within_fn,
                )
                res = simulate(orders, conv_assign, loading_seq)

                all_results.append({
                    "tote_rule": tote_name,
                    "conv_rule": conv_name,
                    "within_rule": within_name,
                    "makespan": res["makespan"],
                    "total_completion": res["total_completion"],
                    "avg_completion": res["avg_completion"],
                    "tote_sequence": tote_seq,
                    "conv_assignment": conv_assign,
                    "loading_sequence": loading_seq,
                })

    # Rank by makespan first, then by total completion, then avg completion
    all_results.sort(key=lambda r: (r["makespan"], r["total_completion"], r["avg_completion"]))
    for i, r in enumerate(all_results, 1):
        r["rank"] = i

    best = all_results[0]
    second_best = all_results[1] if len(all_results) > 1 else None

    best_plan = {
        "tote_rule": best["tote_rule"],
        "conv_rule": best["conv_rule"],
        "within_rule": best["within_rule"],
        "tote_sequence": best["tote_sequence"],
        "conv_assignment": best["conv_assignment"],
        "tote_contents": tote_contents,
        "loading_sequence": best["loading_sequence"],
        "baseline_res": baseline_res,
    }

    second_plan = None
    if second_best:
        second_plan = {
            "tote_rule": second_best["tote_rule"],
            "conv_rule": second_best["conv_rule"],
            "within_rule": second_best["within_rule"],
            "tote_sequence": second_best["tote_sequence"],
            "conv_assignment": second_best["conv_assignment"],
            "tote_contents": tote_contents,
            "loading_sequence": second_best["loading_sequence"],
            "baseline_res": baseline_res,
        }

    return baseline_plan, best_plan, second_plan, all_results


# ==========================
# Printing helpers
# ==========================
def _print_plan(label: str, orders: List[Order], plan: Dict[str, Any], res: Dict[str, Any]) -> None:
    base = plan["baseline_res"]
    delta = (res["makespan"] - base["makespan"]) / base["makespan"] * 100.0 if base["makespan"] else 0.0

    print("\n" + "-" * 90)
    print(label)
    print("-" * 90)
    print(f"Tote={plan['tote_rule']} | ConvAssign={plan['conv_rule']} | WithinTote={plan['within_rule']}")
    print(f"Makespan={res['makespan']}s | TotalCT={res['total_completion']}s | AvgCT={res['avg_completion']:.1f}s")
    print(f"%Δ Makespan vs baseline: {delta:+.1f}%")

    queues = defaultdict(list)
    for o in sorted(orders, key=lambda x: x.id):
        queues[plan["conv_assignment"][o.id]].append(o.id)

    print("\nConveyor queues (conv_num -> order ids in FIFO queue):")
    for c in range(1, NUM_CONVEYORS + 1):
        print(f"  Conveyor {c}: {queues[c]}")

    print("\nTote order:")
    for i, t in enumerate(plan["tote_sequence"], 1):
        cont = plan["tote_contents"][t]
        items = ", ".join(
            f"{SHAPE_NAMES[s]}×{cont[s]}"
            for s in range(NUM_SHAPES)
            if cont[s] > 0
        )
        print(f"  {i:>2}. Tote {t}: {items}")


def _warn_if_shape_cap_exceeded(orders: List[Order]) -> None:
    total = [0] * NUM_SHAPES
    for o in orders:
        dem = order_demand_vector(o)
        for s in range(NUM_SHAPES):
            total[s] += dem[s]

    exceeded = [(SHAPE_NAMES[s], total[s]) for s in range(NUM_SHAPES) if total[s] > MAX_PER_SHAPE_AVAILABLE]
    if exceeded:
        print("\n" + "!" * 90)
        print("WARNING: Your selected TOP_N orders exceed the available items cap (8 per shape).")
        for name, qty in exceeded:
            print(f"  - {name}: need {qty}, cap is {MAX_PER_SHAPE_AVAILABLE}")
        print("Fix: lower TOP_N or choose a subset of orders so each shape total <= 8.")
        print("!" * 90 + "\n")


# ==========================
# MAIN
# ==========================
def main() -> None:
    input_dir = os.path.join("data", "input")
    output_dir = os.path.join("data", "output")

    types_path = os.path.join(input_dir, "order_itemtypes.csv")
    qty_path = os.path.join(input_dir, "order_quantities.csv")
    totes_path = os.path.join(input_dir, "orders_totes.csv")

    os.makedirs(output_dir, exist_ok=True)

    for p in (types_path, qty_path, totes_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}. Put the 3 generator files in this folder.")

    orders = load_orders_first_n(types_path, qty_path, totes_path, TOP_N)
    _warn_if_shape_cap_exceeded(orders)

    baseline_plan, best_plan, second_plan, all_results = run_strategy_search(orders)

    sim_baseline_path = os.path.join(output_dir, "sim_input_BASELINE.csv")
    sim_best_path = os.path.join(output_dir, "sim_input_BEST.csv")
    sim_second_path = os.path.join(output_dir, "sim_input_SECOND_BEST.csv")
    load_baseline_path = os.path.join(output_dir, "loading_plan_BASELINE.csv")
    load_best_path = os.path.join(output_dir, "loading_plan_BEST.csv")
    load_second_path = os.path.join(output_dir, "loading_plan_SECOND_BEST.csv")
    results_path = os.path.join(output_dir, "strategy_results.csv")

    write_sim_input(orders, baseline_plan["conv_assignment"], sim_baseline_path)
    write_sim_input(orders, best_plan["conv_assignment"], sim_best_path)
    if second_plan is not None:
        write_sim_input(orders, second_plan["conv_assignment"], sim_second_path)

    write_loading_plan_csv(baseline_plan["loading_sequence"], load_baseline_path)
    write_loading_plan_csv(best_plan["loading_sequence"], load_best_path)
    if second_plan is not None:
        write_loading_plan_csv(second_plan["loading_sequence"], load_second_path)

    write_strategy_results_csv(all_results, results_path)

    print("\n" + "=" * 90)
    print("MSE 433 — REALISM-ALIGNED CONVEYOR STRATEGY SEARCH")
    print("=" * 90)
    print(f"Orders used: {len(orders)} (rows 0..{len(orders)-1}) | Conveyors: {NUM_CONVEYORS}")
    print(f"TRAVEL_TIME={TRAVEL_TIME}s | LOADING_BELT_TIME={LOADING_BELT_TIME}s | RELEASE_INTERVAL={RELEASE_INTERVAL}s")
    print(f"Strategy combinations tested: {len(all_results)}")
    print(f"COMPARE_CONVEYOR_ONLY={COMPARE_CONVEYOR_ONLY} | FIXED_TOTE_RULE={FIXED_TOTE_RULE} | FIXED_WITHIN_RULE={FIXED_WITHIN_RULE}")
    print("=" * 90)

    _print_plan("PLAN A — BASELINE", orders, baseline_plan, baseline_plan["baseline_res"])
    best_res = simulate(orders, best_plan["conv_assignment"], best_plan["loading_sequence"])
    _print_plan("PLAN B — BEST", orders, best_plan, best_res)

    if second_plan is not None:
        second_res = simulate(orders, second_plan["conv_assignment"], second_plan["loading_sequence"])
        _print_plan("PLAN C — SECOND BEST", orders, second_plan, second_res)

    print("\nSaved files:")
    print(f"  - {sim_baseline_path}")
    print(f"  - {sim_best_path}")
    if second_plan is not None:
        print(f"  - {sim_second_path}")
    print(f"  - {results_path}")
    print(f"  - {load_baseline_path}")
    print(f"  - {load_best_path}")
    if second_plan is not None:
        print(f"  - {load_second_path}")


if __name__ == "__main__":
    main()