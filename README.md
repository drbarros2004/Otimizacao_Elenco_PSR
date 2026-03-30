# Multi-Period Squad Optimization for Flamengo

## Overview

This repository implements a **multi-period football squad optimization model** in **Julia** using **JuMP** and **Gurobi**.

The project combines a deterministic player-evolution pipeline with a mixed-integer optimization model to decide:

- Who should stay in the squad over time
- Who should be bought/sold across transfer windows
- Who should start in each window
- How to preserve budget feasibility while improving sporting performance

The project was developed as a technical/academic operations research case in the PSR context.

## What Is New in This Version

Compared to the earlier baseline, the current version includes:

- External experiment configuration via `config/experiment.toml`
- Explicit **growth potential** map in the data pipeline
- Explicit **wage map** used in budget dynamics
- A richer objective function with:
  - Starter performance
  - Asset/squad value
  - Chemistry bonus
  - Terminal value (final squad + final cash)
- Chemistry/entrosamento temporal dynamics in the MILP
- Soft constraints with slack penalties
- A post-run result analysis script (`analyze_results.jl`)

## Repository Structure

```text
.
├── analyze_results.jl                 # Post-optimization analysis and summaries
├── main.jl                            # Main pipeline + optimization orchestrator
├── Project.toml / Manifest.toml       # Julia dependencies
├── config/
│   ├── experiment.toml                # Simulation, constraints, objective weights
│   └── market_settings.toml           # League reputation and IR multipliers
├── data/
│   ├── raw/
│   │   ├── player_stats.csv
│   │   └── stats_brasileirao_with_correct_teams.csv
│   └── processed/
│       ├── processed_player_data.csv  # Cleaned player base
│       └── player_window_audit.csv    # Unified player-window technical + financial audit
├── output/
│   ├── squad_decisions.csv            # Player-level decisions per window
│   ├── budget_evolution.csv           # Budget path and deficits
│   └── formation_diagnostics.csv      # Tactical scheme constraints and slacks
├── src/
│   ├── data_loader.jl                 # Ingestion, cleaning, map generation, audit export
│   ├── utils.jl                       # Evolution and market multiplier logic
│   └── model.jl                       # Complete JuMP/Gurobi MILP model
├── analysis/
│   ├── plots.py                       # Optional Python plotting utilities
│   ├── streamlit_dashboard.py         # Interactive pitch dashboard (Streamlit)
│   ├── requirements-streamlit.txt     # Python dependencies for dashboard
│   └── viz/                           # Generated figures
└── scraper/                           # Data collection scripts (Python)
```

## Mathematical Model (MILP)

### Decision Variables

For player $j$ and window $t$:

- $x_{j,t} \in \{0,1\}$: player is in squad
- $y_{j,t} \in \{0,1\}$: player is in starting XI
- $buy_{j,t} \in \{0,1\}$: player is bought
- $sell_{j,t} \in \{0,1\}$: player is sold
- $budget_t \ge 0$: cash balance
- $budget\_deficit_t \ge 0$: soft budget violation variable

Chemistry variables are also included for player pairs:

- Pair-in-squad indicator
- Pair-in-starters indicator
- Chemistry score with temporal update and decay

### Core Constraints

1. Squad flow across windows (with purchase/sale timing)
2. Budget recursion with:
   - Purchase costs + transaction overhead
   - Sale proceeds - sale fee
   - Seasonal revenues
   - Wage/signing-related component
3. Squad size bounds (min/max)
4. Tactical formation bounds by configurable position groups (e.g., GK/CB/RB/LB/CM/RW/LW/ST)
5. Exactly 11 starters per window
6. Starter eligibility: $y_{j,t} \le x_{j,t}$
7. Buy/sell exclusivity in a window
8. Chemistry linearization and temporal dynamics

### Objective Function (Current Version)

The model maximizes a weighted combination of:

1. Asset/squad quality score for rostered players
2. Tactical performance score for starters
3. Chemistry bonus
4. Terminal value (final squad market value + final budget)

And penalizes:

- Friction from transfer activity
- Soft-constraint slacks
- Budget deficits (very large penalty)

All key weights and many constraints are configurable in `config/experiment.toml`.

## Data Pipeline

### 1. Ingestion and Cleaning (`src/data_loader.jl`)

- Loads two raw sources:
  - Base SoFIFA-like dataset
  - Curated Brasileirão dataset
- Removes duplicated players by `player_id`
- Removes Brazilian league duplicates from base source to prefer curated rows
- Computes age from DOB
- Maps positions to granular tactical groups (`GK`, `CB`, `RB`, `LB`, `CM`, `RW`, `LW`, `ST`)
- Handles missing values (value, IR, club name)

### 2. Deterministic Projections

Generated maps across windows:

- `ovr_map[(player_id, window)]`
- `value_map[(player_id, window)]`
- `growth_potential_map[(player_id, window)]`
- `wage_map[(player_id, window)]`

### 3. Market Adjustment (`config/market_settings.toml`)

Acquisition cost is adjusted from market value by league reputation and international reputation (IR):

$$
\text{Acquisition Cost} = \text{Market Value} \times \text{Market Multiplier}
$$

Where the multiplier reflects:

- Origin league reputation vs buyer reputation
- IR premium

### 4. Unified Audit Export

The pipeline exports a consolidated technical/financial audit table:

- `data/processed/player_window_audit.csv`

## Configuration

Main experiment parameters are defined in `config/experiment.toml`:

- `simulation.num_windows`
- `optimization.initial_budget`
- `optimization.seasonal_revenue`
- `optimization.initial_squad_strategy`
  - `top_value`
  - `team:<Club Name>` (example: `team:Flamengo`)
- `constraints.*` (squad limits, transaction costs, penalties)
- `formation_catalog.<scheme>.*` exact starter counts by tactical scheme
- `formation_plan.*` active tactical scheme per window
- `objective.*` weights and chemistry/risk parameters

## How to Run

### Prerequisites

- Julia 1.9+
- Gurobi license and installation (required for optimization; pipeline can still run without it)

Install Julia dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Run Full Pipeline + Optimization

```bash
julia --project=. main.jl
```

### Run with a Custom Experiment File

```bash
julia --project=. main.jl --config config/experiment.toml
```

### Behavior When Gurobi Is Not Available

`main.jl` automatically detects whether JuMP/Gurobi can be loaded.

If unavailable, it still runs data processing and exports processed artifacts, then exits gracefully with guidance.

## Analyze Results

After optimization completes:

```bash
julia --project=. analyze_results.jl
```

This script prints:

- Window-by-window squad composition
- Starter quality metrics
- Transfers (bought/sold) and totals
- Budget trajectory and deficits

## Optional Python Visualization

`analysis/plots.py` can generate visualization figures from `player_window_audit.csv`.

Typical dependencies:

```bash
pip install pandas matplotlib seaborn
```

Run from the `analysis` directory:

```bash
python plots.py
```

Generated images are saved to `analysis/viz/`.

## Programmatic Usage (Julia REPL)

```julia
include("main.jl")

# Load configuration
exp_cfg = load_experiment_config("config/experiment.toml")

# Build maps and processed artifacts
(df, ovr_map, value_map, cost_map, growth_potential_map, wage_map) =
    run_pipeline(exp_cfg.num_windows)

# Solve optimization
results = run_optimization(
    df,
    ovr_map,
    value_map,
    cost_map,
    growth_potential_map,
    wage_map;
    initial_budget = exp_cfg.model_params.initial_budget,
    seasonal_revenue = exp_cfg.model_params.seasonal_revenue,
    initial_squad_strategy = exp_cfg.initial_squad_strategy,
    num_windows = exp_cfg.num_windows,
    model_params_override = exp_cfg.model_params
)
```

## Main Output Files

### Data Outputs

- `data/processed/processed_player_data.csv`
- `data/processed/player_window_audit.csv`

### Optimization Outputs

- `output/squad_decisions.csv`
- `output/budget_evolution.csv`
- `output/formation_diagnostics.csv`

### Optional Interactive Dashboard (Streamlit)

Install dashboard dependencies:

```bash
pip install -r analysis/requirements-streamlit.txt
```

Run dashboard:

```bash
streamlit run analysis/streamlit_dashboard.py
```

## Tech Stack

- Julia
- JuMP.jl
- Gurobi.jl
- CSV.jl
- DataFrames.jl
- TOML stdlib
- (Optional) Python + pandas + matplotlib + seaborn for plots

## Author

Daniel Rebouças de Sousa Barros

## License

Academic/professional project. Free to use for educational purposes.

---

Project Status: Functional (deterministic multi-period optimization with chemistry and configurable objectives)
