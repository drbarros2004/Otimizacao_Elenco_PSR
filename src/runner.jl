function run_pipeline(
    num_windows::Int;
    stochastic_config::Union{Nothing,StochasticConfig} = nothing,
    scenario_tree::Union{Nothing,ScenarioTree} = nothing,
    initial_squad_strategy::String = "top_value",
    top_k_players_per_position::Union{Nothing, Int} = nothing,
    domestic_nationalities::Vector{String} = ["Brazilian", "Brazil", "Brasil"],
    unknown_nationality_is_foreign::Bool = true,
    nationality_fallback_to_league::Bool = true
)
    println("🚀 Starting PSR Squad Optimization Pipeline...")
    println("="^50)

    # STEP 1: Data Ingestion & Cleaning
    df = load_and_clean_data()

    protected_player_ids = Int[]
    if startswith(initial_squad_strategy, "team:")
        team_name = replace(initial_squad_strategy, "team:" => "")
        normalized_team_name = strip(lowercase(team_name))
        team_mask = map(name -> !ismissing(name) && strip(lowercase(String(name))) == normalized_team_name, df.club_name)
        protected_player_ids = Int.(df[team_mask, :player_id])
    end

    if !isnothing(top_k_players_per_position)
        before_count = nrow(df)
        df = filter_top_k_players_by_position(
            df,
            top_k_players_per_position;
            protected_player_ids = protected_player_ids,
        )
        after_count = nrow(df)
        println("🔎 Applied top-K filter by position (K=$(top_k_players_per_position)): $before_count -> $after_count players")
    end

    is_foreign_map = build_is_foreign_map(
        df;
        domestic_nationalities = domestic_nationalities,
        unknown_is_foreign = unknown_nationality_is_foreign,
        fallback_to_league = nationality_fallback_to_league,
    )

    # STEP 2: Projections (Deterministic Logic)
    ovr_map, value_map, growth_potential_map = generate_projections(df, num_windows)

    # STEP 3: Market Reputation (Acquisition Costs)
    cost_map = generate_cost_map(df, value_map, num_windows)

    # STEP 4: Wage Map
    wage_map = generate_wage_map(df, ovr_map, num_windows)

    # STEP 5: Unified Analytical Export
    export_analysis(df, ovr_map, value_map, cost_map, num_windows)

    stochastic_bundle = nothing

    if !isnothing(stochastic_config) && stochastic_config.enabled
        if isnothing(scenario_tree)
            error("Stochastic mode is enabled but no scenario tree was provided to run_pipeline.")
        end

        println("\n🌳 Running stochastic node-indexed projection pipeline...")

        node_ovr_map,
        node_value_map,
        node_growth_potential_map,
        starter_allowed_map,
        sell_allowed_map,
        forced_sell_node_map,
        chemistry_multiplier_map = generate_stochastic_projections(
            df,
            scenario_tree,
            stochastic_config
        )

        node_cost_map = generate_cost_map_by_node(df, node_value_map, scenario_tree)
        node_wage_map = generate_wage_map_by_node(df, node_ovr_map, scenario_tree)
        node_position_requirements = generate_node_position_requirements(scenario_tree)

        export_node_analysis(
            df,
            scenario_tree,
            node_ovr_map,
            node_value_map,
            node_cost_map,
            starter_allowed_map,
            sell_allowed_map,
            forced_sell_node_map,
            chemistry_multiplier_map
        )

        stochastic_bundle = (
            tree = scenario_tree,
            ovr_map = node_ovr_map,
            value_map = node_value_map,
            cost_map = node_cost_map,
            growth_potential_map = node_growth_potential_map,
            wage_map = node_wage_map,
            starter_allowed_map = starter_allowed_map,
            sell_allowed_map = sell_allowed_map,
            forced_sell_node_map = forced_sell_node_map,
            chemistry_multiplier_map = chemistry_multiplier_map,
            position_requirements_map = node_position_requirements,
            allow_root_transactions = stochastic_config.allow_root_transactions,
            is_foreign_map = is_foreign_map
        )

        println("✅ Stochastic node-indexed data generated.")
    end

    println("="^50)
    println("✅ Pipeline finished! Unified report saved in 'data/processed/player_window_audit.csv'.")

    return df, ovr_map, value_map, cost_map, growth_potential_map, wage_map, is_foreign_map, stochastic_bundle
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
                          model_params_override::Union{Nothing,ModelParameters} = nothing,
                          is_foreign_map_override::Union{Nothing,Dict{Int,Bool}} = nothing,
                          domestic_nationalities::Vector{String} = ["Brazilian", "Brazil", "Brasil"],
                          unknown_nationality_is_foreign::Bool = true,
                          nationality_fallback_to_league::Bool = true)

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

    is_foreign_map = isnothing(is_foreign_map_override) ? build_is_foreign_map(
        df;
        domestic_nationalities = domestic_nationalities,
        unknown_is_foreign = unknown_nationality_is_foreign,
        fallback_to_league = nationality_fallback_to_league,
    ) : is_foreign_map_override

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
        model_params.formation_by_window,
        is_foreign_map
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
    run_optimization_stochastic(df::DataFrame, stochastic_bundle;
                                initial_budget::Float64=150e6,
                                initial_squad_strategy::String="top_value")

Runs the node-indexed stochastic optimization model and exports node-level outputs.
"""
function run_optimization_stochastic(
    df::DataFrame,
    stochastic_bundle;
    initial_budget::Float64 = 150e6,
    seasonal_revenue::Float64 = 50e6,
    initial_squad_strategy::String = "top_value",
    model_params_override::Union{Nothing,ModelParameters} = nothing,
    is_foreign_map_override::Union{Nothing,Dict{Int,Bool}} = nothing,
    domestic_nationalities::Vector{String} = ["Brazilian", "Brazil", "Brasil"],
    unknown_nationality_is_foreign::Bool = true,
    nationality_fallback_to_league::Bool = true,
    chemistry_affinity_nationalities::Vector{String} = String[
        "Brazilian", "Brazil", "Brasil", "Brasileiro",
        "Argentine", "Argentinian", "Argentina", "Argentino",
        "Portuguese", "Portugal", "Português", "Portugues"
    ],
    chemistry_affinity_initial_chemistry::Float64 = 0.60,
    chemistry_non_affinity_initial_chemistry::Float64 = 0.15,
    chemistry_initial_squad_chemistry::Float64 = 1.0
)
    println("\n" * "="^60)
    println("🌳 STOCHASTIC SQUAD OPTIMIZATION MODULE")
    println("="^60)

    println("🎛️  Configuring model parameters...")

    model_params = isnothing(model_params_override) ? ModelParameters(
        initial_budget = initial_budget,
        seasonal_revenue = seasonal_revenue
    ) : model_params_override

    root_id = stochastic_bundle.tree.root_id
    default_scheme = first(sort(collect(keys(model_params.formation_catalog))))

    root_cost_map = Dict{Tuple{Int, Int}, Float64}(
        (Int(id), 0) => stochastic_bundle.cost_map[(Int(id), root_id)] for id in df.player_id
    )

    println("\n📋 Selecting initial squad (strategy: $initial_squad_strategy)...")
    initial_squad = if startswith(initial_squad_strategy, "team:")
        team_name = replace(initial_squad_strategy, "team:" => "")
        normalized_team_name = strip(lowercase(team_name))
        team_mask = map(name -> !ismissing(name) && strip(lowercase(String(name))) == normalized_team_name, df.club_name)
        team_players = Int.(df[team_mask, :player_id])

        if isempty(team_players)
            @warn "Team '$team_name' not found. Falling back to top_value strategy."
            initial_scheme = get(model_params.formation_by_window, 0, default_scheme)
            initial_formation = model_params.formation_catalog[initial_scheme]
            Int.(select_top_value_squad(df, root_cost_map, 0, initial_budget, initial_formation))
        else
            println("   Selected $(length(team_players)) players from $team_name")
            team_players
        end
    elseif initial_squad_strategy == "top_value"
        initial_scheme = get(model_params.formation_by_window, 0, default_scheme)
        initial_formation = model_params.formation_catalog[initial_scheme]
        Int.(select_top_value_squad(df, root_cost_map, 0, initial_budget, initial_formation))
    else
        error("Invalid initial_squad_strategy: $initial_squad_strategy")
    end

    println("   ✅ Initial squad size: $(length(initial_squad)) players")

    bundle_foreign_map = hasproperty(stochastic_bundle, :is_foreign_map) ? stochastic_bundle.is_foreign_map : nothing
    is_foreign_map = if !isnothing(is_foreign_map_override)
        is_foreign_map_override
    elseif !isnothing(bundle_foreign_map)
        bundle_foreign_map
    else
        build_is_foreign_map(
            df;
            domestic_nationalities = domestic_nationalities,
            unknown_is_foreign = unknown_nationality_is_foreign,
            fallback_to_league = nationality_fallback_to_league,
        )
    end

    initial_chemistry_map = build_initial_chemistry_map(
        df,
        initial_squad;
        affinity_nationalities = chemistry_affinity_nationalities,
        affinity_initial_chemistry = chemistry_affinity_initial_chemistry,
        non_affinity_initial_chemistry = chemistry_non_affinity_initial_chemistry,
        initial_squad_chemistry = chemistry_initial_squad_chemistry,
    )

    # Re-signings should use market onboarding chemistry (not root squad chemistry).
    onboarding_chemistry_map = build_initial_chemistry_map(
        df,
        Int[];
        affinity_nationalities = chemistry_affinity_nationalities,
        affinity_initial_chemistry = chemistry_affinity_initial_chemistry,
        non_affinity_initial_chemistry = chemistry_non_affinity_initial_chemistry,
        initial_squad_chemistry = chemistry_initial_squad_chemistry,
    )

    println("\n📊 Preparing stochastic optimization data...")
    stochastic_data = build_stochastic_model_data(
        df,
        stochastic_bundle,
        initial_squad,
        model_params.formation_catalog,
        is_foreign_map,
        initial_chemistry_map,
        onboarding_chemistry_map
    )

    model = build_stochastic_squad_optimization_model(stochastic_data, model_params, verbose=true)
    results = solve_stochastic_model(model, stochastic_data, model_params, verbose=true)

    println("\n💾 Exporting stochastic results...")
    export_stochastic_results(results, stochastic_data, "output")

    println("\n" * "="^60)
    println("✅ STOCHASTIC OPTIMIZATION COMPLETE!")
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
    df,
    ovr_map,
    value_map,
    cost_map,
    growth_potential_map,
    wage_map,
    is_foreign_map,
    stochastic_bundle = run_pipeline(
        exp_cfg.num_windows,
        stochastic_config = exp_cfg.stochastic_config,
        scenario_tree = exp_cfg.scenario_tree,
        initial_squad_strategy = exp_cfg.initial_squad_strategy,
        top_k_players_per_position = exp_cfg.top_k_players_per_position,
        domestic_nationalities = exp_cfg.domestic_nationalities,
        unknown_nationality_is_foreign = exp_cfg.unknown_nationality_is_foreign,
        nationality_fallback_to_league = exp_cfg.nationality_fallback_to_league,
    )

    # Check if optimization dependencies are available
    println("\n" * "="^60)
    println("⚙️  CHECKING OPTIMIZATION DEPENDENCIES...")
    println("="^60)

    if GUROBI_AVAILABLE
        println("✅ JuMP and Gurobi are available!")

        try
            results = if exp_cfg.stochastic_config.enabled && !isnothing(stochastic_bundle)
                run_optimization_stochastic(
                    df,
                    stochastic_bundle,
                    initial_budget = exp_cfg.model_params.initial_budget,
                    seasonal_revenue = exp_cfg.model_params.seasonal_revenue,
                    initial_squad_strategy = exp_cfg.initial_squad_strategy,
                    model_params_override = exp_cfg.model_params,
                    is_foreign_map_override = is_foreign_map,
                    chemistry_affinity_nationalities = exp_cfg.chemistry_affinity_nationalities,
                    chemistry_affinity_initial_chemistry = exp_cfg.chemistry_affinity_initial_chemistry,
                    chemistry_non_affinity_initial_chemistry = exp_cfg.chemistry_non_affinity_initial_chemistry,
                    chemistry_initial_squad_chemistry = exp_cfg.chemistry_initial_squad_chemistry,
                )
            else
                run_optimization(
                    df, ovr_map, value_map, cost_map, growth_potential_map, wage_map,
                    initial_budget = exp_cfg.model_params.initial_budget,
                    seasonal_revenue = exp_cfg.model_params.seasonal_revenue,
                    initial_squad_strategy = exp_cfg.initial_squad_strategy,
                    num_windows = exp_cfg.num_windows,
                    model_params_override = exp_cfg.model_params,
                    is_foreign_map_override = is_foreign_map
                )
            end

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
        println("   • data/processed/player_window_audit.csv")
        println("\n💡 To run optimization, install Gurobi:")
        println("   1. Get academic license: https://www.gurobi.com/academia/")
        println("   2. Install Gurobi.jl: julia -e 'using Pkg; Pkg.add(\"Gurobi\"); Pkg.build(\"Gurobi\")'")
        println("\n   Then run: julia main.jl")

        return df, ovr_map, value_map, cost_map, growth_potential_map, wage_map, is_foreign_map, stochastic_bundle
    end
end

