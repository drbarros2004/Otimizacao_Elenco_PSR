"""
Scenario tree structures and builders for stochastic optimization.
"""

Base.@kwdef struct StochasticConfig
    enabled::Bool = false
    allow_root_transactions::Bool = false
    branching_by_stage::Vector{Int} = Int[]
    child_probabilities_by_stage::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    ovr_shock_sigma::Float64 = 0.0
    value_shock_sigma::Float64 = 0.0
    chemistry_conflict_probability::Float64 = 0.0
    injury_probability::Float64 = 0.0
    manual_node_events::Dict{Int, Dict{String, Any}} = Dict{Int, Dict{String, Any}}()
end

struct ScenarioNode
    id::Int
    stage::Int
    parent_id::Union{Nothing, Int}
    children_ids::Vector{Int}
    branch_probability::Float64
    cumulative_probability::Float64
    tactical_scheme::String
    position_requirements::Dict{String, Int}
    metadata::Dict{String, Any}
end

struct ScenarioTree
    nodes::Dict{Int, ScenarioNode}
    root_id::Int
    nodes_by_stage::Dict{Int, Vector{Int}}
    leaf_nodes::Vector{Int}
end

function get_effective_tactical_scheme(node::ScenarioNode)::String
    effective = get(node.metadata, "effective_tactical_scheme", nothing)
    if !isnothing(effective)
        return String(effective)
    end

    override = get(node.metadata, "tactical_override", nothing)
    if isnothing(override)
        return node.tactical_scheme
    end
    return String(override)
end

function build_scenario_tree(
    num_windows::Int,
    branching_by_stage::Vector{Int},
    child_probabilities_by_stage::Vector{Vector{Float64}},
    formation_catalog::Dict{String, Dict{String, Int}},
    formation_by_window::Dict{Int, String}
)
    if num_windows < 1
        error("num_windows must be >= 1 for scenario tree construction.")
    end

    if length(branching_by_stage) != num_windows
        error("branching_by_stage must have length num_windows ($num_windows).")
    end

    if length(child_probabilities_by_stage) != num_windows
        error("child_probabilities_by_stage must have length num_windows ($num_windows).")
    end

    for stage in 1:num_windows
        b = branching_by_stage[stage]
        if b < 1
            error("branching_by_stage[$stage] must be >= 1.")
        end

        probs = child_probabilities_by_stage[stage]
        if length(probs) != b
            error("Stage $stage probability vector length ($(length(probs))) must equal branching factor ($b).")
        end

        if any(p -> p < 0.0 || p > 1.0, probs)
            error("Stage $stage probabilities must be in [0,1].")
        end

        if !isapprox(sum(probs), 1.0; atol=1e-9)
            error("Stage $stage probabilities must sum to 1.0.")
        end
    end

    if isempty(formation_catalog)
        error("formation_catalog cannot be empty when building scenario tree.")
    end

    default_scheme = first(sort(collect(keys(formation_catalog))))
    root_scheme = get(formation_by_window, 0, default_scheme)
    if !haskey(formation_catalog, root_scheme)
        error("Root scheme '$root_scheme' is not available in formation_catalog.")
    end

    nodes = Dict{Int, ScenarioNode}()
    nodes_by_stage = Dict{Int, Vector{Int}}(0 => Int[])

    root_id = 1
    root_node = ScenarioNode(
        root_id,
        0,
        nothing,
        Int[],
        1.0,
        1.0,
        root_scheme,
        deepcopy(formation_catalog[root_scheme]),
        Dict{String, Any}()
    )

    nodes[root_id] = root_node
    push!(nodes_by_stage[0], root_id)

    next_node_id = root_id + 1

    for stage in 1:num_windows
        parent_ids = get(nodes_by_stage, stage - 1, Int[])
        nodes_by_stage[stage] = Int[]

        scheme = get(formation_by_window, stage, root_scheme)
        if !haskey(formation_catalog, scheme)
            error("Stage $stage scheme '$scheme' is not available in formation_catalog.")
        end
        pos_requirements = formation_catalog[scheme]

        for parent_id in parent_ids
            parent = nodes[parent_id]
            probs = child_probabilities_by_stage[stage]

            for child_idx in 1:branching_by_stage[stage]
                child_id = next_node_id
                next_node_id += 1

                child_branch_prob = probs[child_idx]
                child_cum_prob = parent.cumulative_probability * child_branch_prob

                child_node = ScenarioNode(
                    child_id,
                    stage,
                    parent_id,
                    Int[],
                    child_branch_prob,
                    child_cum_prob,
                    scheme,
                    deepcopy(pos_requirements),
                    Dict{String, Any}("child_index" => child_idx)
                )

                nodes[child_id] = child_node
                push!(nodes_by_stage[stage], child_id)
                push!(nodes[parent_id].children_ids, child_id)
            end
        end
    end

    leaf_nodes = copy(get(nodes_by_stage, num_windows, Int[]))
    return ScenarioTree(nodes, root_id, nodes_by_stage, leaf_nodes)
end
