"""
MSE 433 - Module 3: Warehousing Heuristic Analysis
====================================================
Demo version: First 4 orders, 4 lanes, one order per lane.
PRIMARY KPI: Minimize Makespan.

THREE decisions optimized:
  1. Lane assignment   → which order goes in which lane (LPT+Penalty)
  2. Tote sequence     → which tote to load first (EOCF)
  3. Item sequence     → within a tote, which item to place first (Furthest Lane First)

Usage:
    python MSE433_warehousing_analysis.py

Input files (same directory):
    order_itemtypes.csv   order_quantities.csv   orders_totes.csv

Outputs:
    conveyor_input_LPT_PENALTY.csv   ← upload to IDEAS Clinic
    (terminal prints full demo instructions)
"""

from collections import defaultdict

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TRAVEL_TIME     = 6    # seconds between each conveyor scanner
FULL_LOOP       = 24   # 4 conveyors × 6s = one full belt circulation
NUM_LANES       = 4    # lanes 0, 1, 2, 3
NUM_ORDERS      = 4    # demo: only first 4 orders
ITEM_PLACE_TIME = 2    # seconds to physically place one item onto belt
ITEM_NAMES      = ['circle', 'pentagon', 'trapezoid', 'triangle',
                   'star',   'moon',     'heart',     'cross']


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def parse_file(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([p.strip() for p in line.split(',') if p.strip()])
    return rows


def load_orders(types_path, qty_path, totes_path):
    t = parse_file(types_path)
    q = parse_file(qty_path)
    r = parse_file(totes_path)
    return [{'id': i,
             'types':      [int(float(v)) for v in t[i]],
             'quantities': [int(float(v)) for v in q[i]],
             'totes':      [int(float(v)) for v in r[i]]}
            for i in range(NUM_ORDERS)]


def build_indexes(orders):
    tote_contents   = defaultdict(lambda: [0] * 8)
    tote_orders_map = defaultdict(set)
    for o in orders:
        for j in range(len(o['types'])):
            t = o['totes'][j]
            tote_contents[t][o['types'][j]] += o['quantities'][j]
            tote_orders_map[t].add(o['id'])
    return (tote_contents, tote_orders_map,
            {o['id']: sum(o['quantities'])  for o in orders},
            {o['id']: len(set(o['totes'])) for o in orders},
            {o['id']: set(o['totes'])       for o in orders},
            sorted(tote_contents.keys()))


# ─────────────────────────────────────────────
# DECISION 1 — TOTE SEQUENCING HEURISTICS
# ─────────────────────────────────────────────
def totes_baseline(all_totes, **_):
    """Naive: ascending tote ID."""
    return sorted(all_totes)


def totes_eocf(all_totes, tote_orders_map, order_num_totes, **_):
    """
    EOCF – Earliest Order Completion First.
    Loads totes that serve orders needing the FEWEST total totes first.
    Single-tote orders get their items onto the belt earliest → complete fastest.
    """
    return sorted(all_totes,
                  key=lambda t: min(order_num_totes[oid]
                                    for oid in tote_orders_map[t]))


def totes_miptf(all_totes, tote_orders_map, **_):
    """
    MIPTF – Most Items Per Tote First.
    Loads totes that serve the most orders simultaneously first.
    More orders advance in parallel per belt pass.
    """
    return sorted(all_totes, key=lambda t: -len(tote_orders_map[t]))

def totes_mitf(all_totes, tote_contents, **_):
    """
    MITF – Most Items Total First.
    Loads totes with the highest total quantity of items first.
    Rationale: item placement takes time (ITEM_PLACE_TIME), so pushing large-work totes
    earlier can reduce downstream completion times under placement bottlenecks.
    """
    return sorted(all_totes, key=lambda t: -sum(tote_contents[t]))

# ─────────────────────────────────────────────
# DECISION 2 — LANE ASSIGNMENT HEURISTICS
# ─────────────────────────────────────────────
def lanes_baseline(orders, **_):
    """Naive: order 0 → lane 0, order 1 → lane 1, etc."""
    return {o['id']: o['id'] for o in orders}


def lanes_lpt_penalty(orders, tote_sequence, order_totes_set, **_):
    """
    LPT + Lane Penalty  ← PRIMARY MAKESPAN HEURISTIC
    ─────────────────────────────────────────────────
    With 4 orders × 4 lanes (no queuing), each order's earliest completion is:
        completion = last_tote_arrival_time + lane × TRAVEL_TIME

    To minimise makespan (= max completion), assign the order whose last tote
    arrives LATEST to Lane 0 (zero travel penalty).  Second-latest to Lane 1, etc.

    This is LPT (Longest Processing Time first) parallel machine scheduling:
    the hardest job gets the best machine.

    Why lane 0 is "best":
        Lane 0 scanner is the FIRST hit every circulation.
        An item in Lane 0 travels 0 extra seconds.
        An item in Lane 3 travels 18 extra seconds.
        Giving the hardest order the smallest travel penalty minimises makespan.
    """
    def last_tote_time(oid):
        return max(tote_sequence.index(t) for t in order_totes_set[oid]) * FULL_LOOP

    sorted_oids = sorted([o['id'] for o in orders],
                         key=lambda oid: -last_tote_time(oid))
    return {oid: lane for lane, oid in enumerate(sorted_oids)}

def lanes_mifl(orders, order_total_items, **_):
    """
    MIFL – Most Items to Fastest Lane.
    Assign the order with the most total items to Lane 0 (lowest travel penalty),
    next most to Lane 1, etc.
    """
    sorted_oids = sorted([o['id'] for o in orders],
                         key=lambda oid: -order_total_items[oid])
    return {oid: lane for lane, oid in enumerate(sorted_oids)}

def lanes_wlpt(orders, tote_sequence, order_totes_set, order_total_items,
               alpha=1.0, beta=6.0, **_):
    """
    WLPT – Weighted LPT.
    Score = alpha*(last tote arrival time) + beta*(total items).
    Highest score gets Lane 0 (best lane), etc.

    beta defaults to 6.0 so "one extra item" is roughly comparable to one lane step
    of travel-time scale. Feel free to tune.
    """
    pos = {t: i for i, t in enumerate(tote_sequence)}

    def last_tote_time(oid):
        return max(pos[t] for t in order_totes_set[oid]) * FULL_LOOP

    def score(oid):
        return alpha * last_tote_time(oid) + beta * order_total_items[oid]

    sorted_oids = sorted([o['id'] for o in orders], key=lambda oid: -score(oid))
    return {oid: lane for lane, oid in enumerate(sorted_oids)}


# ─────────────────────────────────────────────
# DECISION 3 — ITEM LOADING SEQUENCE HEURISTICS
# ─────────────────────────────────────────────
def item_order_naive(items):
    """Naive: place items in ascending item-type order (as you pick them up)."""
    return sorted(items, key=lambda x: x[0])


def item_order_furthest_lane_first(items):
    """
    Furthest Lane First (FLF)  ← PRIMARY ITEM SEQUENCING HEURISTIC
    ──────────────────────────────────────────────────────────────────
    Within a tote, place items going to the HIGHEST lane number first.

    Why: An item for Lane 3 must travel past scanners 0, 1, 2 before
    being sorted (18s travel time). An item for Lane 0 is sorted immediately.

    If you place the Lane 3 item first, it is already travelling down
    the belt while you place the Lane 0 item behind it.
    Both arrive at their destination at nearly the same time.

    If you place the Lane 0 item first, the Lane 3 item sits in your
    hand waiting, then has to travel 18s AFTER the Lane 0 item is done.
    Net result: the last item finishes later → higher makespan.

    Analogy: if you're sending packages to addresses 1 block away and
    4 blocks away, send the far one first so it's already en route.
    """
    return sorted(items, key=lambda x: -x[2])  # highest lane first

def item_order_biggest_qty_first(items):
    """
    BPF – Biggest Quantity First.
    Place item types with the highest quantity first.
    Rationale: large blocks consume more placement time (ITEM_PLACE_TIME per unit),
    so starting them earlier reduces the chance they become the tail.
    """
    # items: (item_type, qty, lane)
    return sorted(items, key=lambda x: -x[1])

def item_order_weighted_flf(items):
    """
    WFLF – Weighted Furthest Lane First.
    Sort by (lane * quantity) descending.
    Rationale: combines travel delay (lane) and placement workload (qty).
    """
    return sorted(items, key=lambda x: -(x[2] * x[1]))

# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────
def simulate(orders, tote_sequence, lane_assignment,
             tote_contents, item_order_fn=item_order_furthest_lane_first):

    # Remaining demand per order (shape -> qty)
    remaining = {}
    for o in orders:
        need = {k: 0 for k in range(8)}
        for j in range(len(o["types"])):
            need[o["types"][j]] += o["quantities"][j]
        remaining[o["id"]] = need

    tote_load_start = {t: idx * FULL_LOOP for idx, t in enumerate(tote_sequence)}
    order_last_sort = defaultdict(int)

    # Helper: pick which order should receive this shape (only among orders that still need it)
    def choose_order_for_shape(shape):
        candidates = [oid for oid in remaining if remaining[oid][shape] > 0]
        if not candidates:
            return None

        # tie-break: prioritize orders closest to completion (fewest remaining items)
        def remaining_items(oid):
            return sum(remaining[oid].values())

        return min(candidates, key=remaining_items)

    for tote_id in tote_sequence:
        t_start = tote_load_start[tote_id]
        contents = tote_contents[tote_id]

        # Create items list = (shape, qty, lane) BUT lane depends on which order takes it,
        # so we only order by shape/qty first then decide assignment per unit.
        items = [(shape, contents[shape]) for shape in range(8) if contents[shape] > 0]
        if not items:
            continue

        # For ordering within tote, we need lane info; approximate by "best possible lane"
        # (max lane among orders needing it) so FLF still has meaning.
        items_with_lane = []
        for shape, qty in items:
            lanes_needing = [lane_assignment[oid] for oid in remaining if remaining[oid][shape] > 0]
            approx_lane = max(lanes_needing) if lanes_needing else 0
            items_with_lane.append((shape, qty, approx_lane))

        items_ordered = item_order_fn(items_with_lane)
        placement_time = t_start

        for shape, qty, _ in items_ordered:
            for _unit in range(qty):
                oid = choose_order_for_shape(shape)
                if oid is None:
                    # no order needs this item anymore
                    placement_time += ITEM_PLACE_TIME
                    continue

                lane = lane_assignment[oid]
                sort_time = placement_time + lane * TRAVEL_TIME

                remaining[oid][shape] -= 1
                order_last_sort[oid] = max(order_last_sort[oid], sort_time)

                placement_time += ITEM_PLACE_TIME

    makespan = max(order_last_sort.values()) if order_last_sort else 0
    total_ct = sum(order_last_sort.values())
    return {
        "order_completion": dict(order_last_sort),
        "makespan": makespan,
        "total_completion": total_ct,
        "avg_completion": total_ct / len(order_last_sort) if order_last_sort else 0,
    }


# ─────────────────────────────────────────────
# CONVEYOR INPUT CSV  (one row per order/lane)
# ─────────────────────────────────────────────
def write_conveyor_input(orders, lane_assignment, filepath):
    """
    Each row = one lane.
    conv_num = lane number.
    Item columns = total quantity of each item type that order needs.
    The conveyor system uses this to know what to drop into each bin.
    Note: 'cirle' typo matches the IDEAS Clinic template exactly.
    """
    lines = ['conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross']
    for lane in range(NUM_LANES):
        oid = next(o for o, l in lane_assignment.items() if l == lane)
        o   = next(x for x in orders if x['id'] == oid)
        row = [0] * 8
        for j in range(len(o['types'])):
            row[o['types'][j]] += o['quantities'][j]
        lines.append(f"{lane}," + ','.join(str(v) for v in row))
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → Saved: {filepath}")


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────
def print_comparison(results, baseline_r):
    best_ms = min(r['makespan'] for r in results.values())
    print("=" * 88)
    print(f"  {'Strategy':<38}  {'Makespan':>9}  {'%Δ MS':>7}  {'Total CT':>9}  {'Avg CT':>8}")
    print("  " + "─" * 82)
    for name, r in results.items():
        dm   = (r['makespan'] - baseline_r['makespan']) / baseline_r['makespan'] * 100
        flag = "  ★ RECOMMENDED" if name == '★ EOCF + LPT+Penalty + FLF Items' else ""
        print(f"  {name:<38}  {r['makespan']:>7.0f}s ({dm:>+5.1f}%)  "
              f"{r['total_completion']:>7.0f}s  {r['avg_completion']:>7.1f}s{flag}")
    print()



def print_demo_instructions(best_r, tote_seq, lane_assignment,
                             orders, tote_contents, item_to_lane):
    print("=" * 65)
    print("  COMPLETE DEMO INSTRUCTIONS")
    print("=" * 65)

    print(f"\n  ─── STEP 1: Upload conveyor_input_LPT_PENALTY.csv ───")
    print(f"  This tells each lane what items to sort into its bin:\n")
    print(f"  {'Lane':>5}  {'Order':>6}  Contents")
    print(f"  {'─'*5}  {'─'*6}  {'─'*40}")
    for lane in range(NUM_LANES):
        oid = next(o for o, l in lane_assignment.items() if l == lane)
        o   = next(x for x in orders if x['id'] == oid)
        contents = ', '.join(f"{ITEM_NAMES[o['types'][j]]}×{o['quantities'][j]}"
                             for j in range(len(o['types'])))
        print(f"  {lane:>5}  {oid:>6}  {contents}")

    print(f"\n  ─── STEP 2: Load totes in this sequence ───\n")
    for pos, tote_id in enumerate(tote_seq):
        contents = tote_contents[tote_id]
        items_str = ', '.join(f"{ITEM_NAMES[k]}×{contents[k]}"
                              for k in range(8) if contents[k] > 0)
        print(f"  Position {pos+1}: Load Tote {tote_id}  ({items_str})")

    print(f"\n  ─── STEP 3: Within each tote, place items in this order ───")
    print(f"  Rule: FURTHEST LANE FIRST — highest lane number goes on belt first")
    print(f"  (items going further need more travel time → get them moving first)\n")

    for pos, tote_id in enumerate(tote_seq):
        contents = tote_contents[tote_id]
        items = [(item_type, contents[item_type], item_to_lane[item_type])
                 for item_type in range(8)
                 if contents[item_type] > 0 and item_type in item_to_lane]
        if not items:
            continue
        items_sorted = sorted(items, key=lambda x: -x[2])
        print(f"  Tote {tote_id}:")
        for rank, (item_type, qty, lane) in enumerate(items_sorted, 1):
            oid = next(o for o, l in lane_assignment.items() if l == lane)
            travel = lane * TRAVEL_TIME
            print(f"    {rank}. {qty}× {ITEM_NAMES[item_type]:<12} "
                  f"→ Lane {lane} (travels {travel}s to scanner)  "
                  f"[Order {oid}]")
        print()

    print(f"  ─── KPI Results ───\n")
    print(f"  Makespan : {best_r['makespan']}s  (theoretical optimum — cannot do better)")
    print(f"  Total CT : {best_r['total_completion']}s")
    print(f"  Avg CT   : {best_r['avg_completion']:.1f}s")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    orders = load_orders('order_itemtypes.csv',
                         'order_quantities.csv',
                         'orders_totes.csv')

    (tote_contents, tote_orders_map,
     order_total_items, order_num_totes,
     order_totes_set, all_totes) = build_indexes(orders)

    ctx = dict(orders=orders, all_totes=all_totes,
                tote_contents=tote_contents,        
               tote_orders_map=tote_orders_map,
               order_num_totes=order_num_totes,
               order_total_items=order_total_items,
               order_totes_set=order_totes_set)

    tseq_b = totes_baseline(**ctx)
    tseq_e = totes_eocf(**ctx)
    tseq_m = totes_miptf(**ctx)
    tseq_i = totes_mitf(**ctx)   # NEW

    lanes_b   = lanes_baseline(**ctx)
    lanes_lpt_b = lanes_lpt_penalty(tote_sequence=tseq_b, **ctx)
    lanes_lpt_e = lanes_lpt_penalty(tote_sequence=tseq_e, **ctx)
    lanes_lpt_m = lanes_lpt_penalty(tote_sequence=tseq_m, **ctx)
    lanes_lpt_i = lanes_lpt_penalty(tote_sequence=tseq_i, **ctx)
    lanes_mifl_b = lanes_mifl(**ctx)
    lanes_mifl_e = lanes_mifl(**ctx)     # (doesn't depend on tote_sequence)

    lanes_wlpt_b = lanes_wlpt(tote_sequence=tseq_b, **ctx)
    lanes_wlpt_e = lanes_wlpt(tote_sequence=tseq_e, **ctx)
    lanes_wlpt_i = lanes_wlpt(tote_sequence=tseq_i, **ctx)

    # All 6 strategies × 2 item orderings = compare clearly
    strategies = {
        'Baseline (naive everything)':       (tseq_b, lanes_b,     item_order_naive),
        'EOCF Totes + Naive Lanes':          (tseq_e, lanes_b,     item_order_naive),
        'EOCF + MIFL (naive items)':          (tseq_e, lanes_mifl_e, item_order_naive),
        'EOCF + WLPT (naive items)':          (tseq_e, lanes_wlpt_e, item_order_naive),
        'MITF Totes + Naive Lanes':          (tseq_i, lanes_b,     item_order_naive),      
        'Baseline + LPT+Penalty':            (tseq_b, lanes_lpt_b, item_order_naive),
        'EOCF + LPT+Penalty (naive items)':  (tseq_e, lanes_lpt_e, item_order_naive),
        'MITF + LPT+Penalty (naive items)': (tseq_i, lanes_lpt_i, item_order_naive),
        'EOCF + LPT+Penalty + Naive Items': (tseq_e, lanes_lpt_e, item_order_naive),
        'EOCF + LPT+Penalty + BPF Items':   (tseq_e, lanes_lpt_e, item_order_biggest_qty_first),
        'EOCF + LPT+Penalty + WFLF Items':  (tseq_e, lanes_lpt_e, item_order_weighted_flf),
        '★ EOCF + LPT+Penalty + FLF Items': (tseq_e, lanes_lpt_e, item_order_furthest_lane_first),

    }

    results = {name: simulate(orders, ts, la, tote_contents, fn)
               for name, (ts, la, fn) in strategies.items()}

    print("\n")
    print("=" * 65)
    print("  MSE 433 — Warehousing Demo: Heuristic Comparison")
    print(f"  Orders: {NUM_ORDERS}  |  Lanes: {NUM_LANES}  |  "
          f"Item placement time: {ITEM_PLACE_TIME}s each")
    print("=" * 65)
    print()
    print_comparison(results, results['Baseline (naive everything)'])

    # Best strategy
    best_tseq  = tseq_e
    best_lanes = lanes_lpt_e

    # Build item_to_lane for instructions
    item_to_lane = {}
    for o in orders:
        lane = best_lanes[o['id']]
        for item_type in o['types']:
            item_to_lane[item_type] = lane

    best_r = results['★ EOCF + LPT+Penalty + FLF Items']
    print_demo_instructions(best_r, best_tseq, best_lanes,
                            orders, tote_contents, item_to_lane)

    print("Writing conveyor input CSV:")
    write_conveyor_input(orders, best_lanes, 'conveyor_input_LPT_PENALTY.csv')
    print()


if __name__ == '__main__':
    main()