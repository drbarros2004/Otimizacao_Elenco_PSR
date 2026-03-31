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

# Monetary normalization used inside optimization models.
# Internal model unit is million EUR to improve numerical conditioning.
const MONEY_SCALE = 1e6

money_to_millions(value::Real) = Float64(value) / MONEY_SCALE
millions_to_money(value::Real) = Float64(value) * MONEY_SCALE

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
