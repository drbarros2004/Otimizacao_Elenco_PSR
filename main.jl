using CSV, DataFrames, Dates, TOML

# Try to load optimization dependencies
const GUROBI_AVAILABLE = try
    using JuMP, Gurobi
    true
catch e
    false
end

# --- 1. Import Dependencies ---
include("src/data_loader.jl")
include("src/model.jl")

# --- 2. Configuration ---
# You can change experiment parameters in config/experiment.toml
const DEFAULT_EXPERIMENT_CONFIG_PATH = "config/experiment.toml"

struct ExperimentConfig
    num_windows::Int
    initial_squad_strategy::String
    model_params::ModelParameters
end

"""
    _parse_formation_counts(formation_cfg::Dict{String,Any})

Parses exact formation counts from TOML into Dict{String, Int}.
Expected format: POSITION = count (e.g., GK = 1, CB = 2).
"""
function _parse_formation_counts(formation_cfg::Dict{String,Any})
    formation = Dict{String, Int}()

    if isempty(formation_cfg)
        error("Section [formation] cannot be empty in experiment config.")
    end

    for (raw_pos, raw_count) in formation_cfg
        pos = strip(String(raw_pos))
        if isempty(pos)
            error("Formation entry has an empty position key.")
        end

        if !(raw_count isa Integer)
            error("Formation '$pos' must be an integer starter count.")
        end

        count = Int(raw_count)
        if count < 0
            error("Invalid formation count for '$pos': $count.")
        end
        formation[pos] = count
    end

    total = sum(values(formation))
    if total != 11
        error("Invalid formation: exact starter counts sum to $total, but XI must be exactly 11.")
    end

    return formation
end

"""
    _parse_window_key(raw_key::AbstractString)

Parses a window key from TOML. Accepted formats: "0", "w0", "window_0".
"""
function _parse_window_key(raw_key::AbstractString)
    key = lowercase(strip(String(raw_key)))

    if occursin(r"^\d+$", key)
        return parse(Int, key)
    elseif occursin(r"^w\d+$", key)
        return parse(Int, key[2:end])
    elseif occursin(r"^window_\d+$", key)
        return parse(Int, split(key, "_")[end])
    else
        error("Invalid formation_plan key '$raw_key'. Use one of: '0', 'w0', 'window_0'.")
    end
end

"""
    _parse_tactical_strategy(cfg::Dict{String,Any}, num_windows::Int)

Parses deterministic tactical setup:
1. `formation_catalog.<scheme>.<position> = count`
2. `formation_plan.<window> = <scheme>`

Backwards compatibility:
- If `formation_catalog` is missing, falls back to legacy `[formation]` as scheme `default`.
"""
function _parse_tactical_strategy(cfg::Dict{String,Any}, num_windows::Int)
    formation_catalog_cfg = get(cfg, "formation_catalog", Dict{String,Any}())
    legacy_formation_cfg = get(cfg, "formation", Dict{String,Any}())
    formation_plan_cfg = get(cfg, "formation_plan", Dict{String,Any}())

    formation_catalog = Dict{String, Dict{String, Int}}()

    if !isempty(formation_catalog_cfg)
        for (raw_scheme, raw_limits) in formation_catalog_cfg
            scheme = strip(String(raw_scheme))
            if isempty(scheme)
                error("Found empty scheme name in [formation_catalog].")
            end
            if !(raw_limits isa Dict)
                error("Scheme '$scheme' in [formation_catalog] must be a table of positions.")
            end

            counts = _parse_formation_counts(raw_limits)
            formation_catalog[scheme] = counts
        end
    elseif !isempty(legacy_formation_cfg)
        formation_catalog["default"] = _parse_formation_counts(legacy_formation_cfg)
    else
        error("You must provide either [formation_catalog] or [formation] in experiment config.")
    end

    if isempty(formation_catalog)
        error("No valid tactical scheme found in experiment config.")
    end

    formation_by_window = Dict{Int, String}()

    if isempty(formation_plan_cfg)
        default_scheme = first(sort(collect(keys(formation_catalog))))
        for w in 0:num_windows
            formation_by_window[w] = default_scheme
        end
    else
        for (raw_window, raw_scheme) in formation_plan_cfg
            w = _parse_window_key(String(raw_window))
            if w < 0 || w > num_windows
                error("formation_plan defines window $w, but simulation windows are 0:$num_windows.")
            end

            scheme = strip(String(raw_scheme))
            if !haskey(formation_catalog, scheme)
                error("formation_plan references unknown scheme '$scheme'.")
            end

            formation_by_window[w] = scheme
        end

        # Fill unspecified windows with the most recent known scheme.
        default_scheme = get(formation_by_window, 0, first(sort(collect(keys(formation_catalog)))))
        for w in 0:num_windows
            if haskey(formation_by_window, w)
                default_scheme = formation_by_window[w]
            else
                formation_by_window[w] = default_scheme
            end
        end
    end

    return formation_catalog, formation_by_window
end

"""
    load_experiment_config(config_path::String=DEFAULT_EXPERIMENT_CONFIG_PATH)

Loads experiment settings from TOML and validates key constraints.
"""
function load_experiment_config(config_path::String=DEFAULT_EXPERIMENT_CONFIG_PATH)
    if !isfile(config_path)
        error("Experiment config not found at '$config_path'.")
    end

    cfg = TOML.parsefile(config_path)

    sim_cfg = get(cfg, "simulation", Dict{String,Any}())
    opt_cfg = get(cfg, "optimization", Dict{String,Any}())
    constraints_cfg = get(cfg, "constraints", Dict{String,Any}())
    objective_cfg = get(cfg, "objective", Dict{String,Any}())

    num_windows = Int(get(sim_cfg, "num_windows", 4))
    if num_windows < 1
        error("simulation.num_windows must be >= 1.")
    end

    initial_budget = Float64(get(opt_cfg, "initial_budget", 150e6))
    seasonal_revenue = Float64(get(opt_cfg, "seasonal_revenue", 50e6))
    initial_squad_strategy = String(get(opt_cfg, "initial_squad_strategy", "top_value"))

    max_squad_size = Int(get(constraints_cfg, "max_squad_size", 30))
    min_squad_size = Int(get(constraints_cfg, "min_squad_size", 18))
    friction_penalty = Float64(get(constraints_cfg, "friction_penalty", 1.5))
    transaction_cost_buy = Float64(get(constraints_cfg, "transaction_cost_buy", 0.12))
    transaction_cost_sell = Float64(get(constraints_cfg, "transaction_cost_sell", 0.10))
    signing_bonus_rate = Float64(get(constraints_cfg, "signing_bonus_rate", 0.5))
    salary_cap_multiplier_initial = Float64(get(constraints_cfg, "salary_cap_multiplier_initial", 1.2))
    salary_cap_window_factor = Float64(get(constraints_cfg, "salary_cap_window_factor", 1.0))
    salary_cap_penalty = Float64(get(constraints_cfg, "salary_cap_penalty", 1e5))
    if min_squad_size > max_squad_size
        error("constraints.min_squad_size cannot be greater than constraints.max_squad_size.")
    end
    if initial_budget <= 0
        error("optimization.initial_budget must be > 0.")
    end
    if seasonal_revenue < 0
        error("optimization.seasonal_revenue must be >= 0.")
    end
    if salary_cap_multiplier_initial <= 0
        error("constraints.salary_cap_multiplier_initial must be > 0.")
    end
    if salary_cap_window_factor <= 0
        error("constraints.salary_cap_window_factor must be > 0.")
    end
    if salary_cap_penalty <= 0
        error("constraints.salary_cap_penalty must be > 0.")
    end

    formation_catalog, formation_by_window = _parse_tactical_strategy(cfg, num_windows)

    weight_quality = Float64(get(objective_cfg, "weight_quality", 0.80))
    weight_potential = Float64(get(objective_cfg, "weight_potential", 0.15))
    decay_quimica = Float64(get(objective_cfg, "decay_quimica", 0.70))
    peso_asset = Float64(get(objective_cfg, "peso_asset", 0.2))
    peso_performance = Float64(get(objective_cfg, "peso_performance", 1.0))
    bonus_entrosamento = Float64(get(objective_cfg, "bonus_entrosamento", 2.0))
    risk_appetite = Float64(get(objective_cfg, "risk_appetite", 1.0))

    model_params = ModelParameters(
        initial_budget = initial_budget,
        seasonal_revenue = seasonal_revenue,
        max_squad_size = max_squad_size,
        min_squad_size = min_squad_size,
        friction_penalty = friction_penalty,
        transaction_cost_buy = transaction_cost_buy,
        transaction_cost_sell = transaction_cost_sell,
        signing_bonus_rate = signing_bonus_rate,
        salary_cap_multiplier_initial = salary_cap_multiplier_initial,
        salary_cap_window_factor = salary_cap_window_factor,
        salary_cap_penalty = salary_cap_penalty,
        formation_catalog = formation_catalog,
        formation_by_window = formation_by_window,
        weight_quality = weight_quality,
        weight_potential = weight_potential,
        decay_quimica = decay_quimica,
        peso_asset = peso_asset,
        peso_performance = peso_performance,
        bonus_entrosamento = bonus_entrosamento,
        risk_appetite = risk_appetite
    )

    println("🧪 Loaded experiment config: $config_path")
    println("   Windows: $num_windows")
    println("   Initial strategy: $initial_squad_strategy")
    println("   Tactical schemes: $(join(sort(collect(keys(formation_catalog))), ", "))")
    println("   Budget: €$(round(initial_budget/1e6, digits=1))M | Seasonal revenue: €$(round(seasonal_revenue/1e6, digits=1))M")
    println("   Salary cap multiplier (initial payroll): x$(round(salary_cap_multiplier_initial, digits=2))")
    println("   Salary cap window factor: $(round(salary_cap_window_factor, digits=2)) | penalty: $(round(salary_cap_penalty, digits=1))")

    return ExperimentConfig(num_windows, initial_squad_strategy, model_params)
end

"""
    get_experiment_config_path(args::Vector{String})

Reads optional CLI arg: --config <path>
"""
function get_experiment_config_path(args::Vector{String})
    for i in 1:length(args)
        if args[i] == "--config"
            if i == length(args)
                error("Missing value after --config.")
            end
            return args[i + 1]
        end
    end
    return DEFAULT_EXPERIMENT_CONFIG_PATH
end

function run_pipeline(num_windows::Int)
    println("🚀 Starting PSR Squad Optimization Pipeline...")
    println("="^50)

    # STEP 1: Data Ingestion & Cleaning
    df = load_and_clean_data()

    # STEP 2: Projections (Deterministic Logic)
    ovr_map, value_map, growth_potential_map = generate_projections(df, num_windows)

    # STEP 3: Market Reputation (Acquisition Costs)
    cost_map = generate_cost_map(df, value_map, num_windows)

    # STEP 5: Wage Map
    wage_map = generate_wage_map(df, ovr_map, num_windows)

    # STEP 6: Unified Analytical Export
    export_analysis(df, ovr_map, value_map, cost_map, num_windows)

    println("="^50)
    println("✅ Pipeline finished! Master report saved in 'data/processed/master_audit.csv'.")

    return df, ovr_map, value_map, cost_map, growth_potential_map, wage_map
end

"""
    run_optimization(df::DataFrame, ovr_map::Dict, value_map::Dict, cost_map::Dict,
                     growth_potential_map::Dict, wage_map::Dict;
                     initial_budget::Float64=150e6, initial_squad_strategy::String="top_value")

Runs the squad optimization model.

# Arguments
- `df::DataFrame`: Player master data
- `ovr_map, value_map, cost_map, growth_potential_map, wage_map`: Projection dictionaries
- `initial_budget::Float64`: Starting budget in euros (default: 150M)
- `seasonal_revenue::Float64`: Revenue per season (default: 50M)
- `initial_squad_strategy::String`: How to select initial squad
"""
function run_optimization(df::DataFrame, ovr_map::Dict, value_map::Dict, cost_map::Dict,
                          growth_potential_map::Dict, wage_map::Dict;
                          initial_budget::Float64 = 150e6,
                          seasonal_revenue::Float64 = 50e6,
                          initial_squad_strategy::String = "top_value",
                          num_windows::Int = 4,
                          model_params_override::Union{Nothing,ModelParameters} = nothing)

    println("\n" * "="^60)
    println("⚽ SQUAD OPTIMIZATION MODULE")
    println("="^60)

    # -------------------------------------------------------------------------
    # 1. Configure Model Parameters
    # -------------------------------------------------------------------------
    println("🎛️  Configuring model parameters...")

    model_params = isnothing(model_params_override) ? ModelParameters(
        initial_budget = initial_budget,
        seasonal_revenue = seasonal_revenue
    ) : model_params_override

    # -------------------------------------------------------------------------
    # 2. Select Initial Squad
    # -------------------------------------------------------------------------
    println("\n📋 Selecting initial squad (strategy: $initial_squad_strategy)...")

    windows = 0:num_windows
    initial_window = first(windows)
    default_scheme = first(sort(collect(keys(model_params.formation_catalog))))

    initial_squad = if startswith(initial_squad_strategy, "team:")
        # Extract players from specific team
        team_name = replace(initial_squad_strategy, "team:" => "")
        normalized_team_name = strip(lowercase(team_name))
        team_mask = map(name -> !ismissing(name) && strip(lowercase(String(name))) == normalized_team_name, df.club_name)
        team_players = df[team_mask, :player_id]

        if isempty(team_players)
            @warn "Team '$team_name' not found. Falling back to top_value strategy."
            initial_scheme = get(model_params.formation_by_window, initial_window, default_scheme)
            initial_formation = model_params.formation_catalog[initial_scheme]
            select_top_value_squad(df, cost_map, initial_window, initial_budget, initial_formation)
        else
            println("   Selected $(length(team_players)) players from $team_name")
            team_players
        end
    elseif initial_squad_strategy == "top_value"
        initial_scheme = get(model_params.formation_by_window, initial_window, default_scheme)
        initial_formation = model_params.formation_catalog[initial_scheme]
        select_top_value_squad(df, cost_map, initial_window, initial_budget, initial_formation)
    else
        error("Invalid initial_squad_strategy: $initial_squad_strategy")
    end

    println("   ✅ Initial squad size: $(length(initial_squad)) players")

    # -------------------------------------------------------------------------
    # 3. Prepare Model Data
    # -------------------------------------------------------------------------
    println("\n📊 Preparing optimization data...")

    model_data = ModelData(
        df,
        windows,
        ovr_map,
        value_map,
        cost_map,
        growth_potential_map,
        wage_map,
        initial_squad,
        model_params.formation_catalog,
        model_params.formation_by_window
    )

    # -------------------------------------------------------------------------
    # 4. Build and Solve Model
    # -------------------------------------------------------------------------
    model = build_squad_optimization_model(model_data, model_params, verbose=true)
    results = solve_model(model, model_data, model_params, verbose=true)

    # -------------------------------------------------------------------------
    # 5. Export Results
    # -------------------------------------------------------------------------
    println("\n💾 Exporting results...")
    export_results(results, model_data, "output")

    println("\n" * "="^60)
    println("✅ OPTIMIZATION COMPLETE!")
    println("="^60)

    return results
end

"""
    select_top_value_squad(df::DataFrame, cost_map::Dict, window::Int, budget::Float64, formation::Dict{String, Int})

Selects the best value-for-money squad within budget constraints using a greedy heuristic.
Ensures tactical balance using the configured formation groups.
"""
function select_top_value_squad(
    df::DataFrame,
    cost_map::Dict,
    window::Int,
    budget::Float64,
    formation::Dict{String, Int}
)
    println("   Using greedy heuristic to select initial squad...")

    formation_positions = collect(keys(formation))
    valid_position = Set(formation_positions)

    min_requirements = Dict{String, Int}()
    max_per_position = Dict{String, Int}()
    for (pos, starters_required) in formation
        # Keep at least one backup by default.
        min_requirements[pos] = max(1, starters_required + 1)
        max_per_position[pos] = max(min_requirements[pos], starters_required + 4)
    end

    # Create value/cost ratio metric
    candidates = DataFrame(
        player_id = df.player_id,
        name = df.name,
        pos_group = df.pos_group,
        ovr = df.overall_rating,
        cost = [get(cost_map, (id, window), Inf) for id in df.player_id]
    )

    # Remove unaffordable players
    candidates = candidates[candidates.cost .<= budget, :]

    # Calculate value metric (OVR / sqrt(cost)) - favors good players at reasonable prices
    candidates.value_metric = candidates.ovr ./ sqrt.(candidates.cost ./ 1e6)
    sort!(candidates, :value_metric, rev=true)

    # Greedy selection with position constraints
    selected = Int[]
    current_cost = 0.0
    position_counts = Dict(pos => 0 for pos in formation_positions)

    for row in eachrow(candidates)
        pos = row.pos_group
        cost = row.cost

        if !(pos in valid_position)
            continue
        end

        # Check if we can afford and need this position
        if current_cost + cost <= budget * 0.85 &&  # Keep 15% buffer
           position_counts[pos] < max_per_position[pos] &&
           length(selected) < 25

            push!(selected, row.player_id)
            current_cost += cost
            position_counts[pos] += 1
        end

        # Stop if we have a balanced squad
        if all(position_counts[k] >= min_requirements[k] for k in keys(min_requirements)) &&
           length(selected) >= 18
            break
        end
    end

    # Validate squad composition
    for (pos, min_count) in min_requirements
        if position_counts[pos] < min_count
            @warn "Initial squad has only $(position_counts[pos]) $pos players (minimum: $min_count)"
        end
    end

    composition = join(sort(["$pos=$(position_counts[pos])" for pos in keys(position_counts)]), ", ")
    println("   Squad composition: $composition")
    println("   Total cost: €$(round(current_cost/1e6, digits=1))M / €$(round(budget/1e6, digits=1))M")

    return selected
end

"""
    main()

Main entry point - runs both data pipeline and optimization.
"""
function main()
    config_path = get_experiment_config_path(ARGS)
    exp_cfg = load_experiment_config(config_path)

    # Run data pipeline
    df, ovr_map, value_map, cost_map, growth_potential_map, wage_map = run_pipeline(exp_cfg.num_windows)

    # Check if optimization dependencies are available
    println("\n" * "="^60)
    println("⚙️  CHECKING OPTIMIZATION DEPENDENCIES...")
    println("="^60)

    if GUROBI_AVAILABLE
        println("✅ JuMP and Gurobi are available!")

        try
            # Run optimization
            results = run_optimization(
                df, ovr_map, value_map, cost_map, growth_potential_map, wage_map,
                initial_budget = exp_cfg.model_params.initial_budget,
                seasonal_revenue = exp_cfg.model_params.seasonal_revenue,
                initial_squad_strategy = exp_cfg.initial_squad_strategy,
                num_windows = exp_cfg.num_windows,
                model_params_override = exp_cfg.model_params
            )

            println("\n" * "="^60)
            println("🎉 PIPELINE COMPLETE!")
            println("="^60)
            println("\nTo analyze results, run:")
            println("   julia analyze_results.jl")

            return df, results
        catch e
            println("\n❌ Error during optimization: $e")
            println("\nStacktrace:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
            rethrow(e)
        end
    else
        println("\n⚠️  Gurobi not available!")
        println("\n📊 Data pipeline completed successfully.")
        println("   Generated files:")
        println("   • data/processed/processed_player_data.csv")
        println("   • data/processed/master_audit.csv")
        println("\n💡 To run optimization, install Gurobi:")
        println("   1. Get academic license: https://www.gurobi.com/academia/")
        println("   2. Install Gurobi.jl: julia -e 'using Pkg; Pkg.add(\"Gurobi\"); Pkg.build(\"Gurobi\")'")
        println("\n   Then run: julia main.jl")

        return df, ovr_map, value_map, cost_map, growth_potential_map, wage_map
    end
end

# Check if the script is run directly from the terminal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
