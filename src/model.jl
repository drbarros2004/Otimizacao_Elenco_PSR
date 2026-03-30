"""
Squad Optimization Model (Multi-Period MILP) - COMPLETE VERSION
PSR Technical Case - Flamengo Squad Management

EXACT translation of Python/PuLP model including:
- Chemistry/Entrosamento system with temporal dynamics
- Growth potential in objective function
- Terminal value (elenco + caixa final)
- Soft constraints with slack variables
- Timing: purchases at t-1 affect squad at t
"""

using JuMP, Gurobi, DataFrames, CSV

# Declaring constants
const S_MAX = 10.0         
const S_INICIAL = 5.0      
const BONUS_INCREMENTO = 0.5  
const DECAY_QUIMICA_DEFAULT = 0.70  

const PESO_ASSET = 0.2      
const PESO_PERFORMANCE = 1.0  
const BONUS_ENTROSAMENTO = 2.0  

const BIG_M = 1000.0

const P_POSICAO = 1e6
const P_SALARIO = 1e5
const P_CAIXA = 1e9

# Declaring our model structures
struct ModelData
    players::DataFrame
    windows::UnitRange{Int}
    ovr_map::Dict{Tuple{Int,Int}, Int}
    value_map::Dict{Tuple{Int,Int}, Float64}
    cost_map::Dict{Tuple{Int,Int}, Float64}
    growth_potential_map::Dict{Tuple{Int,Int}, Float64}
    wage_map::Dict{Tuple{Int,Int}, Float64}
    initial_squad::Vector{Int}
    formation_catalog::Dict{String, Dict{String, Int}}
    formation_by_window::Dict{Int, String}
end

struct ModelParameters
    initial_budget::Float64
    seasonal_revenue::Float64
    max_squad_size::Int
    min_squad_size::Int
    friction_penalty::Float64
    transaction_cost_buy::Float64
    transaction_cost_sell::Float64
    signing_bonus_rate::Float64
    formation_catalog::Dict{String, Dict{String, Int}}
    formation_by_window::Dict{Int, String}
    weight_quality::Float64
    weight_potential::Float64
    decay_quimica::Float64
    peso_asset::Float64
    peso_performance::Float64
    bonus_entrosamento::Float64
    risk_appetite::Float64

    function ModelParameters(;
        initial_budget::Float64 = 100e6,
        seasonal_revenue::Float64 = 50e6,
        max_squad_size::Int = 30,
        min_squad_size::Int = 18,
        friction_penalty::Float64 = 1.5,
        transaction_cost_buy::Float64 = 0.12,
        transaction_cost_sell::Float64 = 0.10,
        signing_bonus_rate::Float64 = 0.5,
        formation_catalog::Dict{String, Dict{String, Int}} = Dict(
            "default" => Dict(
                "GK" => 1,
                "CB" => 2,
                "RB" => 1,
                "LB" => 1,
                "CM" => 3,
                "RW" => 1,
                "LW" => 1,
                "ST" => 1
            )
        ),
        formation_by_window::Dict{Int, String} = Dict(0 => "default"),
        weight_quality::Float64 = 0.80,
        weight_potential::Float64 = 0.15,
        decay_quimica::Float64 = DECAY_QUIMICA_DEFAULT,
        peso_asset::Float64 = PESO_ASSET,
        peso_performance::Float64 = PESO_PERFORMANCE,
        bonus_entrosamento::Float64 = BONUS_ENTROSAMENTO,
        risk_appetite::Float64 = 1.0
    )
        new(initial_budget, seasonal_revenue, max_squad_size, min_squad_size,
            friction_penalty, transaction_cost_buy, transaction_cost_sell, signing_bonus_rate,
            formation_catalog, formation_by_window,
            weight_quality, weight_potential, decay_quimica,
            peso_asset, peso_performance, bonus_entrosamento, risk_appetite)
    end
end

struct ModelResults
    model::Model
    objective_value::Float64
    solve_time::Float64
    squad_decisions::DataFrame
    budget_evolution::DataFrame
    formation_diagnostics::DataFrame
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function generate_player_pairs(initial_squad::Vector{Int})
    pairs = Tuple{Int,Int}[]
    n = length(initial_squad)
    for i in 1:n
        for j in (i+1):n
            push!(pairs, (initial_squad[i], initial_squad[j]))
        end
    end
    return pairs
end

function _default_scheme_name(params::ModelParameters)
    if !isempty(params.formation_by_window)
        min_window = minimum(collect(keys(params.formation_by_window)))
        return params.formation_by_window[min_window]
    end
    return first(sort(collect(keys(params.formation_catalog))))
end

function get_window_formation(params::ModelParameters, t::Int)
    scheme = get(params.formation_by_window, t, _default_scheme_name(params))
    if !haskey(params.formation_catalog, scheme)
        error("Window $t references unknown tactical scheme '$scheme'.")
    end
    return scheme, params.formation_catalog[scheme]
end

# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

function build_squad_optimization_model(data::ModelData, params::ModelParameters; verbose::Bool=true)
    verbose && println("\n🏗️  Building Complete Multi-Period Squad Optimization Model...")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", 600)  
    set_optimizer_attribute(model, "MIPGap", 0.01)    
    !verbose && set_optimizer_attribute(model, "OutputFlag", 0)

    # Extract data
    players = data.players
    J = players.player_id
    T = data.windows
    pos_groups = Dict(row.player_id => row.pos_group for row in eachrow(players))
    formation_positions = String[]
    for limits in values(params.formation_catalog)
        append!(formation_positions, collect(keys(limits)))
    end
    formation_positions = unique(sort(formation_positions))

    if isempty(formation_positions)
        error("No tactical positions found in formation catalog.")
    end

    # Generate pairs for chemistry
    pairs = generate_player_pairs(data.initial_squad)
    verbose && println("  ├─ Chemistry pairs: $(length(pairs))")

    # ==== DECISION VARIABLES ====
    verbose && println("  ├─ Creating decision variables...")

    @variable(model, x[j in J, t in T], Bin)
    @variable(model, y[j in J, t in T], Bin)
    @variable(model, buy[j in J, t in T], Bin)
    @variable(model, sell[j in J, t in T], Bin)
    @variable(model, budget[t in T] >= 0)
    @variable(model, budget_deficit[t in T] >= 0)

    # Chemistry variables
    @variable(model, ParElenco[(i,j) in pairs, t in T], Bin)
    @variable(model, ParTitular[(i,j) in pairs, t in T], Bin)
    @variable(model, 0 <= Quimica[(i,j) in pairs, t in T] <= S_MAX)

    # Soft constraint slacks
    @variable(model, slack_posicao[p in formation_positions, t in T] >= 0)
    @variable(model, slack_salario[t in T] >= 0)

    # ==== INITIAL CONDITIONS ====
    verbose && println("  ├─ Initial conditions...")

    @constraint(model, budget[first(T)] == params.initial_budget)

    for j in J
        if j in data.initial_squad
            @constraint(model, x[j, first(T)] == 1)
            @constraint(model, buy[j, first(T)] == 0)
            @constraint(model, sell[j, first(T)] == 0)
        else
            @constraint(model, x[j, first(T)] == 0)
        end
    end

    # Initial chemistry for base pairs
    for (i, j) in pairs
        if i in data.initial_squad && j in data.initial_squad
            @constraint(model, Quimica[(i,j), first(T)] == S_INICIAL)
        else
            @constraint(model, Quimica[(i,j), first(T)] == 0)
        end
    end

    # ==== SQUAD FLOW (timing: buy at t-1 affects squad at t) ====
    verbose && println("  ├─ Squad flow constraints (Python timing)...")

    for j in J, t in T
        if t > first(T)
            @constraint(model, x[j, t] == x[j, t-1] + buy[j, t-1] - sell[j, t-1])
        end
    end

    # ==== BUDGET DYNAMICS ====
    verbose && println("  ├─ Budget constraints...")

    for t in T
        if t > first(T)
            revenue = (t % 2 == 0) ? params.seasonal_revenue : 0.0

            @constraint(model,
                budget[t] == budget[t-1]
                    - sum(
                        (1 + params.transaction_cost_buy) * data.cost_map[(j,t)] * buy[j,t-1] +
                        data.wage_map[(j,t)] * 52 * params.signing_bonus_rate * buy[j,t-1]
                        for j in J
                    )
                    + sum((1 - params.transaction_cost_sell) * data.value_map[(j,t)] * sell[j,t-1] for j in J)
                    + revenue
                    + budget_deficit[t]
            )
        end
    end

    # ==== SQUAD SIZE ====
    verbose && println("  ├─ Squad size constraints...")

    for t in T
        @constraint(model, sum(x[j,t] for j in J) <= params.max_squad_size)
        @constraint(model, sum(x[j,t] for j in J) >= params.min_squad_size)
    end

    # ==== FORMATION (EXACT) ====
    verbose && println("  ├─ Tactical formation constraints (exact)...")

    for t in T
        @constraint(model, sum(y[j,t] for j in J) == 11)

        _, formation_t = get_window_formation(params, t)

        for (pos, required_count) in formation_t
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            @constraint(model,
                sum(y[j,t] for j in players_in_pos) == required_count
            )
        end
    end

    # ==== STARTER ELIGIBILITY ====
    verbose && println("  ├─ Starter eligibility...")

    for j in J, t in T
        @constraint(model, y[j,t] <= x[j,t])
    end

    # ==== TRANSACTION EXCLUSIVITY ====
    verbose && println("  ├─ Transaction exclusivity...")

    for j in J, t in T
        @constraint(model, buy[j,t] + sell[j,t] <= 1)
    end

    # ==== CHEMISTRY LINEARIZATION ====
    verbose && println("  ├─ Chemistry linearization...")

    for (i, j) in pairs, t in T
        # ParElenco = x[i] × x[j]
        @constraint(model, ParElenco[(i,j), t] <= x[i, t])
        @constraint(model, ParElenco[(i,j), t] <= x[j, t])
        @constraint(model, ParElenco[(i,j), t] >= x[i,t] + x[j,t] - 1)

        # ParTitular = y[i] × y[j]
        @constraint(model, ParTitular[(i,j), t] <= y[i, t])
        @constraint(model, ParTitular[(i,j), t] <= y[j, t])
        @constraint(model, ParTitular[(i,j), t] >= y[i,t] + y[j,t] - 1)
    end

    # ==== CHEMISTRY DYNAMICS ====
    verbose && println("  ├─ Chemistry temporal dynamics...")

    for (i, j) in pairs, t in T
        if t > first(T)
            M = S_MAX

            # Case 1: Together → chemistry grows
            @constraint(model,
                Quimica[(i,j), t] <= Quimica[(i,j), t-1] + BONUS_INCREMENTO + M * (1 - ParElenco[(i,j), t])
            )
            @constraint(model,
                Quimica[(i,j), t] >= Quimica[(i,j), t-1] + BONUS_INCREMENTO - M * (1 - ParElenco[(i,j), t])
            )

            # Case 2: Separated → chemistry decays
            @constraint(model,
                Quimica[(i,j), t] <= params.decay_quimica * Quimica[(i,j), t-1] + M * ParElenco[(i,j), t]
            )
            @constraint(model,
                Quimica[(i,j), t] >= params.decay_quimica * Quimica[(i,j), t-1] - M * ParElenco[(i,j), t]
            )
        end
    end

    # ==== OBJECTIVE FUNCTION ====
    verbose && println("  ├─ Building objective function...")

    obj_terms = AffExpr(0.0)

    # 1. Score de Mercado (Asset Value) - applies to ALL squad
    for j in J, t in T
        if t > first(T)  # Skip t=0
            score_mercado = (
                params.weight_quality * data.ovr_map[(j,t)] +
                params.weight_potential * data.growth_potential_map[(j,t)]
            )
            add_to_expression!(obj_terms, params.peso_asset * score_mercado * x[j,t])
        end
    end

    # 2. Score Tático (Performance) - applies to STARTERS
    for j in J, t in T
        if t > first(T)
            score_tatico = data.ovr_map[(j,t)] * 1.0
            add_to_expression!(obj_terms, params.peso_performance * score_tatico * y[j,t])
        end
    end

    # 3. Chemistry bonus
    for (i, j) in pairs, t in T
        if t > first(T)
            add_to_expression!(obj_terms, params.bonus_entrosamento * Quimica[(i,j), t])
        end
    end

    # 4. Friction penalty
    for j in J, t in T
        if t > first(T) && t > 1  # buy at t-1 affects t
            add_to_expression!(obj_terms, -params.friction_penalty * buy[j, t-1])
        end
    end

    # 5. Soft constraint penalties
    for t in T
        add_to_expression!(obj_terms, -P_SALARIO * slack_salario[t])
        add_to_expression!(obj_terms, -P_CAIXA * budget_deficit[t])
    end

    # 6. Terminal value
    t_final = last(T)
    valor_elenco_final = sum(data.value_map[(j, t_final)] * x[j, t_final] for j in J)
    add_to_expression!(obj_terms, 1.0 * valor_elenco_final)
    add_to_expression!(obj_terms, params.risk_appetite * budget[t_final])

    @objective(model, Max, obj_terms)

    verbose && println("  └─ ✅ Model construction complete!")
    verbose && println("\n📊 Model Statistics:")
    verbose && println("    • Players: $(length(J))")
    verbose && println("    • Windows: $(length(T))")
    verbose && println("    • Chemistry Pairs: $(length(pairs))")
    verbose && println("    • Variables: $(num_variables(model))")
    verbose && println("    • Constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

    return model
end

# =============================================================================
# SOLVING
# =============================================================================

function solve_model(model::Model, data::ModelData, params::ModelParameters; verbose::Bool=true)
    verbose && println("\n🚀 Solving optimization model...")
    verbose && println("="^60)

    start_time = time()
    optimize!(model)
    solve_time = time() - start_time

    status = termination_status(model)
    verbose && println("\n📌 Termination Status: $status")

    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT]
        error("Model did not solve successfully. Status: $status")
    end

    obj_value = objective_value(model)
    verbose && println("🎯 Objective Value: $(round(obj_value, digits=2))")
    verbose && println("⏱️  Solve Time: $(round(solve_time, digits=2))s")

    verbose && println("\n📦 Extracting solution...")

    J = data.players.player_id
    T = data.windows
    pos_groups = Dict(row.player_id => row.pos_group for row in eachrow(data.players))

    decision_rows = []
    for j in J, t in T
        x_val = round(Int, value(model[:x][j, t]))
        y_val = round(Int, value(model[:y][j, t]))
        b_val = round(Int, value(model[:buy][j, t]))
        s_val = round(Int, value(model[:sell][j, t]))

        if x_val == 1 || b_val == 1 || s_val == 1
            formation_scheme = get(data.formation_by_window, t, "default")
            push!(decision_rows, (
                player_id = j,
                window = t,
                formation_scheme = formation_scheme,
                in_squad = x_val,
                is_starter = y_val,
                bought = b_val,
                sold = s_val,
                ovr = data.ovr_map[(j, t)],
                market_value = data.value_map[(j, t)],
                acquisition_cost = data.cost_map[(j, t)]
            ))
        end
    end

    df_decisions = DataFrame(decision_rows)

    budget_rows = [(window = t, cash_balance = value(model[:budget][t]),
                    deficit = value(model[:budget_deficit][t])) for t in T]
    df_budget = DataFrame(budget_rows)

    diagnostics_rows = []
    for t in T
        scheme_name = get(data.formation_by_window, t, "default")
        if !haskey(data.formation_catalog, scheme_name)
            error("Missing scheme '$scheme_name' for window $t in formation catalog.")
        end

        counts = data.formation_catalog[scheme_name]
        for (pos, required_count) in counts
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            starters_count = sum(round(Int, value(model[:y][j, t])) for j in players_in_pos)

            push!(diagnostics_rows, (
                window = t,
                formation_scheme = scheme_name,
                pos_group = pos,
                required_count = required_count,
                actual_starters = starters_count,
                slack_titular = 0.0
            ))
        end
    end

    df_formation_diagnostics = DataFrame(diagnostics_rows)

    verbose && println("✅ Solution extracted successfully!")

    return ModelResults(model, obj_value, solve_time, df_decisions, df_budget, df_formation_diagnostics)
end

# =============================================================================
# EXPORT
# =============================================================================

function export_results(results::ModelResults, data::ModelData, output_dir::String="output")
    mkpath(output_dir)

    df_output = leftjoin(results.squad_decisions,
                         data.players[:, [:player_id, :name, :pos_group]],
                         on=:player_id)

    CSV.write("$output_dir/squad_decisions.csv", df_output)
    CSV.write("$output_dir/budget_evolution.csv", results.budget_evolution)
    CSV.write("$output_dir/formation_diagnostics.csv", results.formation_diagnostics)

    println("\n💾 Results exported to '$output_dir/':")
    println("   • squad_decisions.csv")
    println("   • budget_evolution.csv")
    println("   • formation_diagnostics.csv")

    return df_output
end
