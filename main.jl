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
# You can change these values here without touching the internal logic
NUM_WINDOWS = 6  # 3 years of simulation (2 windows per year)

function run_pipeline()
    println("🚀 Starting PSR Squad Optimization Pipeline...")
    println("="^50)

    # STEP 1: Data Ingestion & Cleaning
    df = load_and_clean_data()

    # STEP 2: Projections (Deterministic Logic)
    ovr_map, value_map = generate_projections(df, NUM_WINDOWS)

    # STEP 3: Market Reputation (Acquisition Costs)
    cost_map = generate_cost_map(df, value_map, NUM_WINDOWS)

    # STEP 5: Wage Map
    wage_map = generate_wage_map(df, ovr_map, NUM_WINDOWS)

    # STEP 6: Unified Analytical Export
    export_analysis(df, ovr_map, value_map, cost_map, NUM_WINDOWS)

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
                          num_windows::Int = NUM_WINDOWS)

    println("\n" * "="^60)
    println("⚽ SQUAD OPTIMIZATION MODULE")
    println("="^60)

    # -------------------------------------------------------------------------
    # 1. Select Initial Squad
    # -------------------------------------------------------------------------
    println("\n📋 Selecting initial squad (strategy: $initial_squad_strategy)...")

    windows = 0:num_windows
    initial_window = first(windows)

    initial_squad = if startswith(initial_squad_strategy, "team:")
        # Extract players from specific team
        team_name = replace(initial_squad_strategy, "team:" => "")
        team_players = df[df.club_name .== team_name, :player_id]

        if isempty(team_players)
            @warn "Team '$team_name' not found. Falling back to top_value strategy."
            select_top_value_squad(df, cost_map, initial_window, initial_budget)
        else
            println("   Selected $(length(team_players)) players from $team_name")
            team_players
        end
    elseif initial_squad_strategy == "top_value"
        select_top_value_squad(df, cost_map, initial_window, initial_budget)
    else
        error("Invalid initial_squad_strategy: $initial_squad_strategy")
    end

    println("   ✅ Initial squad size: $(length(initial_squad)) players")

    # -------------------------------------------------------------------------
    # 2. Prepare Model Data
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
        initial_squad
    )

    # -------------------------------------------------------------------------
    # 3. Configure Model Parameters
    # -------------------------------------------------------------------------
    println("🎛️  Configuring model parameters...")

    model_params = ModelParameters(
        initial_budget = initial_budget,
        seasonal_revenue = seasonal_revenue,
        max_squad_size = 30,
        min_squad_size = 18,
        friction_penalty = 1.5,  # Equilibrado
        transaction_cost_buy = 0.12,
        transaction_cost_sell = 0.10,
        signing_bonus_rate = 0.5,
        formation = Dict(
            "GK"  => (1, 1),
            "DEF" => (3, 5),
            "MID" => (3, 5),
            "FWD" => (1, 3)
        ),
        # Strategy weights (Equilibrado)
        weight_quality = 0.80,
        weight_potential = 0.15,
        decay_quimica = 0.70,
        peso_asset = 0.2,
        peso_performance = 1.0,
        bonus_entrosamento = 2.0,
        risk_appetite = 1.0
    )

    # -------------------------------------------------------------------------
    # 4. Build and Solve Model
    # -------------------------------------------------------------------------
    model = build_squad_optimization_model(model_data, model_params, verbose=true)
    results = solve_model(model, model_data, verbose=true)

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
    select_top_value_squad(df::DataFrame, cost_map::Dict, window::Int, budget::Float64)

Selects the best value-for-money squad within budget constraints using a greedy heuristic.
Ensures tactical balance (GK, DEF, MID, FWD).
"""
function select_top_value_squad(df::DataFrame, cost_map::Dict, window::Int, budget::Float64)
    println("   Using greedy heuristic to select initial squad...")

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
    position_counts = Dict("GK" => 0, "DEF" => 0, "MID" => 0, "FWD" => 0)

    # Minimum requirements for a valid squad
    min_requirements = Dict("GK" => 2, "DEF" => 6, "MID" => 6, "FWD" => 3)
    max_per_position = Dict("GK" => 3, "DEF" => 10, "MID" => 10, "FWD" => 7)

    for row in eachrow(candidates)
        pos = row.pos_group
        cost = row.cost

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

    println("   Squad composition: GK=$(position_counts["GK"]), DEF=$(position_counts["DEF"]), MID=$(position_counts["MID"]), FWD=$(position_counts["FWD"])")
    println("   Total cost: €$(round(current_cost/1e6, digits=1))M / €$(round(budget/1e6, digits=1))M")

    return selected
end

"""
    main()

Main entry point - runs both data pipeline and optimization.
"""
function main()
    # Run data pipeline
    df, ovr_map, value_map, cost_map, growth_potential_map, wage_map = run_pipeline()

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
                initial_budget = 150e6,
                seasonal_revenue = 50e6,
                initial_squad_strategy = "top_value"
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
