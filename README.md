# MSE 433 Module 3: Warehouse Conveyor Heuristics

This repository evaluates conveyor loading heuristics for order fulfillment and exports the files needed for simulation/demo runs.

The main script is `algorithm.py`.

## What the Script Does

`algorithm.py`:

1. Loads order data from 3 generator CSV files.
2. Selects the first `TOP_N` orders (default: 6).
3. Builds tote/item loading sequences using configurable heuristics.
4. Simulates conveyor sorting with FIFO active-order capture logic.
5. Compares strategy combinations and ranks them by:
   - `makespan`
   - `total_completion`
   - `avg_completion`
6. Writes simulation input files and loading plans for baseline, best, and second-best strategies.

## Required Input Files

Place these files in `data/input/`:

- `data/input/order_itemtypes.csv`
- `data/input/order_quantities.csv`
- `data/input/orders_totes.csv`

If any are missing, the script raises a `FileNotFoundError`.

## How to Run

```bash
python algorithm.py
```

Dependencies: Python 3 standard library only.

## Tunable Parameters

Edit these constants near the top of `algorithm.py`:

- `TOP_N = 6`: number of orders used from input rows.
- `NUM_CONVEYORS = 4`: number of conveyor lanes.
- `TRAVEL_TIME = 6`: seconds for an item to move between adjacent conveyors.
- `ITEM_PLACE_TIME = 2`: seconds to place one item.
- `LOADING_BELT_TIME = 5`: observed belt release interval.
- `RELEASE_INTERVAL = max(ITEM_PLACE_TIME, LOADING_BELT_TIME)`.
- `TOTE_CHANGEOVER_TIME = 0`: optional tote swap penalty.

### Strategy Search Mode

By default, the script compares only conveyor assignment while keeping tote and within-tote rules fixed:

- `COMPARE_CONVEYOR_ONLY = True`
- `FIXED_TOTE_RULE = "ID_ASC"`
- `FIXED_WITHIN_RULE = "BPF"`

If you set `COMPARE_CONVEYOR_ONLY = False`, it runs a fuller grid search:

- Tote rules: `ID_ASC`, `MITF`, `EOCF`
- Conveyor rules: `BASELINE_CYCLE`, `LPT_BALANCE`, `LPT_BALANCE_WT`
- Within-tote rules: `NAIVE`, `BPF`, `FLF`, `WFLF`

## Heuristics Implemented

### Tote Sequencing

- `ID_ASC`: totes in ascending ID.
- `MITF`: most items in tote first.
- `EOCF`: totes serving orders with fewer required totes first.

### Conveyor Assignment

- `BASELINE_CYCLE`: order `i` goes to conveyor `(i % NUM_CONVEYORS) + 1`.
- `LPT_BALANCE`: greedy longest-processing-time balancing by estimated order workload.
- `LPT_BALANCE_WT`: complexity-aware workload (items + weighted tote/shape/fragmentation terms).

### Within-Tote Item Ordering

- `NAIVE`: ascending shape ID.
- `BPF`: bigger quantity first.
- `FLF`: farther lane first.
- `WFLF`: weighted farther lane first (`lane * quantity`).

## Simulation Logic

The simulation is event-driven:

- Items are released into conveyor 1 according to the loading plan.
- Items move every `TRAVEL_TIME` seconds to the next conveyor in a loop.
- Each conveyor has a FIFO queue of assigned orders.
- Only the active order at the front can capture matching items.
- Non-matching items continue circulating.

This matches the active-order FIFO capture behavior described in class.

## Output Files

Running `python algorithm.py` generates:

- `data/output/sim_input_BASELINE.csv`
- `data/output/sim_input_BEST.csv`
- `data/output/sim_input_SECOND_BEST.csv` (if a second strategy exists)
- `data/output/loading_plan_BASELINE.csv`
- `data/output/loading_plan_BEST.csv`
- `data/output/loading_plan_SECOND_BEST.csv` (if a second strategy exists)
- `data/output/strategy_results.csv`

### Notes About `sim_input_*.csv`

- Each row corresponds to one selected order.
- The first column is `conv_num`.
- Shape demand columns follow.
- The script currently writes `cirle` as the first shape column name to match existing expected schema in this project.

## Console Output

The script prints:

- baseline plan summary
- best plan summary
- second-best plan summary (if available)
- queue assignments, tote order details, and percentage delta vs baseline
- list of saved files

## Data Cap Warning

The script checks demand against `MAX_PER_SHAPE_AVAILABLE = 8` and prints a warning if selected orders exceed the cap.

## Repository Notes

- Folder layout:
  - `data/input/`: generator input CSVs consumed by `algorithm.py`
  - `data/output/`: latest generated outputs from each run
  - `data/archive/`: older artifacts preserved for reference (including prior `day1` files)
- Re-running the script overwrites files in `data/output/`.
