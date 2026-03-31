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

struct ModelDataStochastic
    players::DataFrame
    tree::ScenarioTree
    node_ids::Vector{Int}
    root_id::Int
    leaf_nodes::Vector{Int}
    nodes_by_stage::Dict{Int, Vector{Int}}
    parent_by_node::Dict{Int, Union{Nothing, Int}}
    stage_by_node::Dict{Int, Int}
    probability_by_node::Dict{Int, Float64}
    path_by_node::Dict{Int, Vector{Int}}
    ovr_node_map::Dict{Tuple{Int, Int}, Int}
    value_node_map::Dict{Tuple{Int, Int}, Float64}
    cost_node_map::Dict{Tuple{Int, Int}, Float64}
    growth_potential_node_map::Dict{Tuple{Int, Int}, Float64}
    wage_node_map::Dict{Tuple{Int, Int}, Float64}
    starter_allowed_map::Dict{Tuple{Int, Int}, Int}
    sell_allowed_map::Dict{Tuple{Int, Int}, Int}
    chemistry_multiplier_map::Dict{Int, Float64}
    position_requirements_map::Dict{Tuple{String, Int}, Int}
    initial_squad::Vector{Int}
    formation_catalog::Dict{String, Dict{String, Int}}
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
    salary_cap_multiplier_initial::Float64
    salary_cap_window_factor::Float64
    salary_cap_penalty::Float64
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
        salary_cap_multiplier_initial::Float64 = 1.2,
        salary_cap_window_factor::Float64 = 1.0,
        salary_cap_penalty::Float64 = P_SALARIO,
        formation_catalog::Dict{String, Dict{String, Int}} = Dict(
            "default" => Dict( # 4-3-3
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
        decay_quimica::Float64 = 0.70,
        peso_asset::Float64 = 0.2,
        peso_performance::Float64 = 1.0,
        bonus_entrosamento::Float64 = 2.0,
        risk_appetite::Float64 = 1.0
    )
        new(initial_budget, seasonal_revenue, max_squad_size, min_squad_size,
            friction_penalty, transaction_cost_buy, transaction_cost_sell, signing_bonus_rate,
            salary_cap_multiplier_initial, salary_cap_window_factor, salary_cap_penalty,
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

struct StochasticModelResults
    model::Model
    objective_value::Float64
    solve_time::Float64
    node_decisions::DataFrame
    budget_evolution::DataFrame
    tree_metadata::DataFrame
    formation_diagnostics::DataFrame
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function _build_node_ancestry_map(
    node_ids::Vector{Int},
    parent_by_node::Dict{Int, Union{Nothing, Int}}
)
    path_by_node = Dict{Int, Vector{Int}}()

    for node_id in node_ids
        path = Int[]
        current = node_id

        while true
            pushfirst!(path, current)
            parent = get(parent_by_node, current, nothing)
            if isnothing(parent)
                break
            end
            current = parent
        end

        path_by_node[node_id] = path
    end

    return path_by_node
end

function build_stochastic_model_data(
    players::DataFrame,
    stochastic_bundle,
    initial_squad::Vector{Int},
    formation_catalog::Dict{String, Dict{String, Int}}
)
    tree = stochastic_bundle.tree
    node_ids = sort(collect(keys(tree.nodes)))

    parent_by_node = Dict{Int, Union{Nothing, Int}}(
        node_id => tree.nodes[node_id].parent_id for node_id in node_ids
    )
    stage_by_node = Dict{Int, Int}(
        node_id => tree.nodes[node_id].stage for node_id in node_ids
    )
    probability_by_node = Dict{Int, Float64}(
        node_id => tree.nodes[node_id].cumulative_probability for node_id in node_ids
    )
    path_by_node = _build_node_ancestry_map(node_ids, parent_by_node)

    model_data = ModelDataStochastic(
        players,
        tree,
        node_ids,
        tree.root_id,
        copy(tree.leaf_nodes),
        deepcopy(tree.nodes_by_stage),
        parent_by_node,
        stage_by_node,
        probability_by_node,
        path_by_node,
        stochastic_bundle.ovr_map,
        stochastic_bundle.value_map,
        stochastic_bundle.cost_map,
        stochastic_bundle.growth_potential_map,
        stochastic_bundle.wage_map,
        stochastic_bundle.starter_allowed_map,
        stochastic_bundle.sell_allowed_map,
        stochastic_bundle.chemistry_multiplier_map,
        stochastic_bundle.position_requirements_map,
        initial_squad,
        formation_catalog
    )

    assert_stochastic_data_contract(model_data)
    return model_data
end

function assert_stochastic_data_contract(data::ModelDataStochastic)
    if isempty(data.node_ids)
        error("Scenario tree has no nodes.")
    end

    if !haskey(data.tree.nodes, data.root_id)
        error("Root node $(data.root_id) not found in scenario tree nodes.")
    end

    for node_id in data.node_ids
        if !haskey(data.parent_by_node, node_id)
            error("Missing parent mapping for node $node_id.")
        end
        if !haskey(data.stage_by_node, node_id)
            error("Missing stage mapping for node $node_id.")
        end
        if !haskey(data.probability_by_node, node_id)
            error("Missing probability mapping for node $node_id.")
        end
        if !haskey(data.path_by_node, node_id)
            error("Missing ancestry path for node $node_id.")
        end

        parent = data.parent_by_node[node_id]
        if isnothing(parent)
            if node_id != data.root_id
                error("Only root node can have no parent. Node $node_id has no parent but root is $(data.root_id).")
            end
        else
            if !haskey(data.tree.nodes, parent)
                error("Node $node_id points to missing parent $parent.")
            end
        end

        node_prob = data.probability_by_node[node_id]
        if node_prob < 0.0 || node_prob > 1.0
            error("Invalid cumulative probability for node $node_id: $node_prob")
        end
    end

    leaf_prob_sum = sum(get(data.probability_by_node, node_id, 0.0) for node_id in data.leaf_nodes)
    if !isapprox(leaf_prob_sum, 1.0; atol=1e-6)
        error("Leaf cumulative probabilities must sum to 1.0. Current sum: $leaf_prob_sum")
    end

    if isempty(data.players)
        error("Player dataset is empty in stochastic model data.")
    end

    player_ids = Int.(data.players.player_id)
    for node_id in data.node_ids
        for p_id in player_ids
            if !haskey(data.ovr_node_map, (p_id, node_id))
                error("Missing ovr_node_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.value_node_map, (p_id, node_id))
                error("Missing value_node_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.cost_node_map, (p_id, node_id))
                error("Missing cost_node_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.growth_potential_node_map, (p_id, node_id))
                error("Missing growth_potential_node_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.wage_node_map, (p_id, node_id))
                error("Missing wage_node_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.starter_allowed_map, (p_id, node_id))
                error("Missing starter_allowed_map entry for player $p_id at node $node_id.")
            end
            if !haskey(data.sell_allowed_map, (p_id, node_id))
                error("Missing sell_allowed_map entry for player $p_id at node $node_id.")
            end
        end

        if !haskey(data.chemistry_multiplier_map, node_id)
            error("Missing chemistry_multiplier_map entry for node $node_id.")
        end

        node = data.tree.nodes[node_id]
        for pos_group in keys(node.position_requirements)
            if !haskey(data.position_requirements_map, (pos_group, node_id))
                error("Missing position_requirements_map entry for position $pos_group at node $node_id.")
            end
        end
    end

    return true
end

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
    verbose && println("\n Building Complete Multi-Period Squad Optimization Model...")

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
            @constraint(model, buy[j, first(T)] == 0)  # Cannot buy in initial window
            @constraint(model, sell[j, first(T)] == 0) # Cannot sell what you don't have
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

            # Calculate signing costs with free agent multiplier
            # Free agents (cost < 100k) demand higher signing bonuses
            signing_costs = AffExpr(0.0)
            for j in J
                cost = data.cost_map[(j, t)]
                wage = data.wage_map[(j, t)]
                ovr = data.ovr_map[(j, t)]

                # Determine if player is a free agent
                is_free_agent = (cost < 100_000)

                if is_free_agent
                    # Apply OVR-based multiplier for free agents
                    multiplier = get_free_agent_signing_multiplier(ovr)
                    signing_cost = wage * 52 * params.signing_bonus_rate * multiplier
                else
                    # Normal signing bonus for regular transfers
                    signing_cost = wage * 52 * params.signing_bonus_rate
                end

                add_to_expression!(signing_costs, signing_cost * buy[j, t-1])
            end

            @constraint(model,
                budget[t] == budget[t-1]
                    - sum((1 + params.transaction_cost_buy) * data.cost_map[(j,t)] * buy[j,t-1] for j in J)
                    - signing_costs
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

    # ==== SALARY CAP (SOFT) ====
    verbose && println("  ├─ Salary cap constraints (soft)...")

    # Baseline payroll comes from the initial squad at window 0.
    initial_window = first(T)
    initial_payroll = sum(data.wage_map[(j, initial_window)] for j in data.initial_squad)
    salary_cap_per_window = initial_payroll * params.salary_cap_multiplier_initial * params.salary_cap_window_factor

    verbose && println("  │  Initial payroll baseline: €$(round(initial_payroll / 1e6, digits=2))M")
    verbose && println("  │  Salary cap per window: €$(round(salary_cap_per_window / 1e6, digits=2))M")

    for t in T
        @constraint(model,
            sum(data.wage_map[(j,t)] * x[j,t] for j in J) <= salary_cap_per_window + slack_salario[t]
        )
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
        add_to_expression!(obj_terms, -params.salary_cap_penalty * slack_salario[t])
        add_to_expression!(obj_terms, -P_CAIXA * budget_deficit[t])
    end

    # 6. Terminal value
    t_final = last(T)
    valor_elenco_final = sum(data.value_map[(j, t_final)] * x[j, t_final] for j in J)
    add_to_expression!(obj_terms, 0.001 * valor_elenco_final)
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

function build_stochastic_squad_optimization_model(
    data::ModelDataStochastic,
    params::ModelParameters;
    verbose::Bool=true
)
    verbose && println("\n Building Stochastic Node-Based Squad Optimization Model...")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", 600)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    !verbose && set_optimizer_attribute(model, "OutputFlag", 0)

    # Extract data
    players = data.players
    J = Int.(players.player_id)
    N = data.node_ids
    root_id = data.root_id
    pos_groups = Dict(Int(row.player_id) => String(row.pos_group) for row in eachrow(players))

    formation_positions = String[]
    for limits in values(data.formation_catalog)
        append!(formation_positions, collect(keys(limits)))
    end
    formation_positions = unique(sort(formation_positions))

    if isempty(formation_positions)
        error("No tactical positions found in stochastic formation catalog.")
    end

    # Chemistry pairs are kept based on initial squad for tractability.
    pairs = generate_player_pairs(data.initial_squad)
    verbose && println("  ├─ Chemistry pairs: $(length(pairs))")

    # ==== DECISION VARIABLES ====
    verbose && println("  ├─ Creating node-indexed decision variables...")

    @variable(model, x[j in J, n in N], Bin)
    @variable(model, y[j in J, n in N], Bin)
    @variable(model, buy[j in J, n in N], Bin)
    @variable(model, sell[j in J, n in N], Bin)
    @variable(model, budget[n in N] >= 0)
    @variable(model, budget_deficit[n in N] >= 0)

    # Chemistry variables
    @variable(model, ParElenco[(i,j) in pairs, n in N], Bin)
    @variable(model, ParTitular[(i,j) in pairs, n in N], Bin)
    @variable(model, 0 <= Quimica[(i,j) in pairs, n in N] <= S_MAX)

    # Soft constraint slacks
    @variable(model, slack_posicao[p in formation_positions, n in N] >= 0)
    @variable(model, slack_salario[n in N] >= 0)

    # ==== INITIAL CONDITIONS (ROOT NODE) ====
    verbose && println("  ├─ Root-node initial conditions...")

    @constraint(model, budget[root_id] == params.initial_budget)

    for j in J
        initial_presence = j in data.initial_squad ? 1 : 0
        @constraint(model, x[j, root_id] == initial_presence)
        # Keep deterministic timing parity: no transaction in initial state.
        @constraint(model, buy[j, root_id] == 0)
        @constraint(model, sell[j, root_id] == 0)
    end

    for (i, j) in pairs
        if i in data.initial_squad && j in data.initial_squad
            @constraint(model, Quimica[(i,j), root_id] == S_INICIAL)
        else
            @constraint(model, Quimica[(i,j), root_id] == 0)
        end
    end

    # ==== SQUAD FLOW (PARENT -> CHILD) ====
    verbose && println("  ├─ Squad flow constraints (parent-child timing)...")

    for n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end

        p = parent::Int
        for j in J
            @constraint(model, x[j, n] == x[j, p] + buy[j, p] - sell[j, p])
        end
    end

    # ==== BUDGET DYNAMICS (PARENT -> CHILD) ====
    verbose && println("  ├─ Budget constraints by node...")

    for n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int

        stage_n = data.stage_by_node[n]
        revenue = (stage_n % 2 == 0) ? params.seasonal_revenue : 0.0

        signing_costs = AffExpr(0.0)
        for j in J
            cost = data.cost_node_map[(j, n)]
            wage = data.wage_node_map[(j, n)]
            ovr = data.ovr_node_map[(j, n)]
            is_free_agent = (cost < 100_000)

            signing_cost = if is_free_agent
                multiplier = get_free_agent_signing_multiplier(ovr)
                wage * 52 * params.signing_bonus_rate * multiplier
            else
                wage * 52 * params.signing_bonus_rate
            end

            add_to_expression!(signing_costs, signing_cost * buy[j, p])
        end

        @constraint(model,
            budget[n] == budget[p]
                - sum((1 + params.transaction_cost_buy) * data.cost_node_map[(j,n)] * buy[j,p] for j in J)
                - signing_costs
                + sum((1 - params.transaction_cost_sell) * data.value_node_map[(j,n)] * sell[j,p] for j in J)
                + revenue
                + budget_deficit[n]
        )
    end

    # ==== SQUAD SIZE ====
    verbose && println("  ├─ Squad size constraints by node...")

    for n in N
        @constraint(model, sum(x[j,n] for j in J) <= params.max_squad_size)
        @constraint(model, sum(x[j,n] for j in J) >= params.min_squad_size)
    end

    # ==== SALARY CAP (SOFT) ====
    verbose && println("  ├─ Salary cap constraints (soft, node-indexed)...")

    initial_payroll = sum(data.wage_node_map[(j, root_id)] for j in data.initial_squad)
    salary_cap_per_node = initial_payroll * params.salary_cap_multiplier_initial * params.salary_cap_window_factor

    verbose && println("  │  Initial payroll baseline: €$(round(initial_payroll / 1e6, digits=2))M")
    verbose && println("  │  Salary cap per node: €$(round(salary_cap_per_node / 1e6, digits=2))M")

    for n in N
        @constraint(model,
            sum(data.wage_node_map[(j,n)] * x[j,n] for j in J) <= salary_cap_per_node + slack_salario[n]
        )
    end

    # ==== FORMATION (NODE-DEPENDENT N_{r,n}) ====
    verbose && println("  ├─ Tactical formation constraints by node (N_{r,n})...")

    for n in N
        @constraint(model, sum(y[j,n] for j in J) == 11)

        for pos in formation_positions
            required_count = get(data.position_requirements_map, (pos, n), 0)
            players_in_pos = [j for j in J if pos_groups[j] == pos]

            @constraint(model,
                sum(y[j,n] for j in players_in_pos) == required_count
            )
        end
    end

    # ==== STARTER ELIGIBILITY + INJURY SANDBOX ====
    verbose && println("  ├─ Starter and sell availability constraints...")

    for j in J, n in N
        starter_allowed = data.starter_allowed_map[(j, n)]
        sell_allowed = data.sell_allowed_map[(j, n)]

        @constraint(model, y[j,n] <= x[j,n] * starter_allowed)
        @constraint(model, sell[j,n] <= sell_allowed)
    end

    # ==== TRANSACTION EXCLUSIVITY ====
    verbose && println("  ├─ Transaction exclusivity by node...")

    for j in J, n in N
        @constraint(model, buy[j,n] + sell[j,n] <= 1)
    end

    # ==== CHEMISTRY LINEARIZATION ====
    verbose && println("  ├─ Chemistry linearization by node...")

    for (i, j) in pairs, n in N
        @constraint(model, ParElenco[(i,j), n] <= x[i, n])
        @constraint(model, ParElenco[(i,j), n] <= x[j, n])
        @constraint(model, ParElenco[(i,j), n] >= x[i,n] + x[j,n] - 1)

        @constraint(model, ParTitular[(i,j), n] <= y[i, n])
        @constraint(model, ParTitular[(i,j), n] <= y[j, n])
        @constraint(model, ParTitular[(i,j), n] >= y[i,n] + y[j,n] - 1)
    end

    # ==== CHEMISTRY DYNAMICS (PARENT -> CHILD) ====
    verbose && println("  ├─ Chemistry parent-child dynamics...")

    for (i, j) in pairs, n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int
        M = S_MAX

        @constraint(model,
            Quimica[(i,j), n] <= Quimica[(i,j), p] + BONUS_INCREMENTO + M * (1 - ParElenco[(i,j), n])
        )
        @constraint(model,
            Quimica[(i,j), n] >= Quimica[(i,j), p] + BONUS_INCREMENTO - M * (1 - ParElenco[(i,j), n])
        )

        @constraint(model,
            Quimica[(i,j), n] <= params.decay_quimica * Quimica[(i,j), p] + M * ParElenco[(i,j), n]
        )
        @constraint(model,
            Quimica[(i,j), n] >= params.decay_quimica * Quimica[(i,j), p] - M * ParElenco[(i,j), n]
        )
    end

    # ==== EXPECTED-VALUE OBJECTIVE (NODE-INDEXED) ====
    verbose && println("  ├─ Building expected-value objective...")

    obj_terms = AffExpr(0.0)

    non_root_nodes = [n for n in N if n != root_id]

    for n in non_root_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        for j in J
            score_mercado = (
                params.weight_quality * data.ovr_node_map[(j,n)] +
                params.weight_potential * data.growth_potential_node_map[(j,n)]
            )
            add_to_expression!(obj_terms, prob_n * params.peso_asset * score_mercado * x[j,n])

            score_tatico = data.ovr_node_map[(j,n)] * 1.0
            add_to_expression!(obj_terms, prob_n * params.peso_performance * score_tatico * y[j,n])

            add_to_expression!(obj_terms, -prob_n * params.friction_penalty * buy[j,n])
        end

        add_to_expression!(obj_terms, -prob_n * params.salary_cap_penalty * slack_salario[n])
        add_to_expression!(obj_terms, -prob_n * P_CAIXA * budget_deficit[n])
    end

    for n in non_root_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        chemistry_multiplier = get(data.chemistry_multiplier_map, n, 1.0)
        for (i, j) in pairs
            add_to_expression!(obj_terms, prob_n * params.bonus_entrosamento * chemistry_multiplier * Quimica[(i,j), n])
        end
    end

    # Terminal value is only accounted for leaf nodes, weighted by cumulative probability.
    for n in data.leaf_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        valor_elenco_final = sum(data.value_node_map[(j, n)] * x[j, n] for j in J)
        add_to_expression!(obj_terms, prob_n * 0.001 * valor_elenco_final)
        add_to_expression!(obj_terms, prob_n * params.risk_appetite * budget[n])
    end

    @objective(model, Max, obj_terms)

    verbose && println("  └─ ✅ Stochastic model construction complete!")
    verbose && println("\n📊 Stochastic Model Statistics:")
    verbose && println("    • Players: $(length(J))")
    verbose && println("    • Nodes: $(length(N))")
    verbose && println("    • Leaf nodes: $(length(data.leaf_nodes))")
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
                starter_in_window = y_val,
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

    player_meta = select(
        data.players,
        :player_id,
        :name,
        :pos_group,
        :club_name,
        :club_league_name
    )
    rename!(player_meta, :club_name => :origin_club, :club_league_name => :origin_league)

    df_output = leftjoin(results.squad_decisions, player_meta, on=:player_id)

    CSV.write("$output_dir/squad_decisions.csv", df_output)
    CSV.write("$output_dir/budget_evolution.csv", results.budget_evolution)
    CSV.write("$output_dir/formation_diagnostics.csv", results.formation_diagnostics)

    println("\n💾 Results exported to '$output_dir/':")
    println("   • squad_decisions.csv")
    println("   • budget_evolution.csv")
    println("   • formation_diagnostics.csv")

    return df_output
end
