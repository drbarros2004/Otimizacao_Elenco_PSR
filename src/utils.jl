using CSV, DataFrames, Dates, Random, Statistics

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

"""
Calculates the player's age based on the Date of Birth (dob).
If the value is missing or invalid, returns a default age of 25.
"""
function calculate_age(dob_value)
    # if missing, return default value
    if ismissing(dob_value) return 25 end
    
    try
        # CSV.read already converted the column to Date
        birth_date = if dob_value isa Date
            dob_value
        else
            # if String, attempt ISO format conversion (yyyy-mm-dd)
            Date(strip(string(dob_value))) 
        end
        
        # reference date 
        today = Date(2026, 03, 26)
        
        # Precise age calculation
        return year(today) - year(birth_date) - (monthday(today) < monthday(birth_date) ? 1 : 0)
    catch e
        # Log warning if parsing fails (helps identifying data quality issues)
        @warn "Failed to calculate age for value: $dob_value. Error: $e"
        return 25
    end
end

"""
Deterministic evolution logic where age directly influences the speed of growth and decay.
"""
function evolution_step(ovr::Float64, pot::Float64, age::Int, value::Float64, window_idx::Int)
    # Age increments every 2 windows
    current_age = (window_idx > 0 && window_idx % 2 != 0) ? age + 1 : age

    # Tunable growth controls
    growth_intercept = 0.34        # Base growth before age discount
    growth_age_slope = 0.0095       # How much growth decreases per age year
    min_growth_coeff = 0.065       # Floor for late growth phase (<30)
    youth_bonus_age_limit = 29     # Applies bonus up to this age (inclusive)
    youth_bonus_multiplier = 1.65  # Extra acceleration for younger players
    
    delta = 0.0
    status = "Stable"

    if current_age < 30
        # Coefficient decreases with age and is bounded by a minimum floor.
        growth_coeff = max(min_growth_coeff, growth_intercept - (growth_age_slope * current_age))

        # Additional youth boost to accelerate early-career development.
        youth_bonus = current_age <= youth_bonus_age_limit ? youth_bonus_multiplier : 1.0
        
        # Growth is proportional to the potential gap
        gap = max(0.0, pot - ovr)
        delta = gap * growth_coeff * youth_bonus
        status = "Growth"
    else
        # Decay accelerates with age. 
        delta = -0.2 * (current_age - 29)
        status = "Decay"
    end
    
    # Update OVR (Rounded to maintain Int, like EA FC)
    new_ovr = clamp(ovr + delta, 40.0, 99.0)
    
    # Update Market Value (Financial Dynamics)
    # Instead of a sharp drop at 28, we use a smooth multiplier 
    # that starts high and crosses 1.0 around age 26-27.
    # Multiplier = 1.2 (young) down to 0.8 (very old)
    age_value_factor = clamp(1.4 - (0.015 * current_age), 0.7, 1.2)
    
    # Value update: accounts for performance delta and natural aging
    new_value = value * (1 + (delta / 100.0)) * age_value_factor
    new_value = max(new_value, 0.5) # Minimum floor of 500k

    return round(new_ovr), new_value, current_age, status
end

"""
criar descricao
"""
function get_market_multiplier(league_name::String, ir_score::Int, market_cfg::Dict)
    buyer_rep = market_cfg["buying_club"]["reputation"]
    default_rep = market_cfg["market_settings"]["default_league_reputation"]
    ir_weight = market_cfg["market_settings"]["ir_multiplier"]
    leagues = market_cfg["leagues"]

    origin_rep = get(leagues, league_name, default_rep)
    gap_premium = max(0.0, (origin_rep - buyer_rep) / buyer_rep)
    ir_premium = (ir_score - 1) * ir_weight

    return 1.0 + gap_premium + ir_premium
end

"""
    infer_free_agent_wages(df::DataFrame) -> Dict{Int, Float64}

Infers realistic wages for free agents based on OVR distribution from top players.
Returns a dictionary mapping OVR -> average wage for that OVR level.

This function analyzes the top 2000 players with valid wages to establish
a baseline wage structure by OVR, which is then used to assign realistic
wages to free agents who have zero or missing wage data.
"""
function infer_free_agent_wages(df::DataFrame)
    println("📊 Inferring realistic wages for free agents from top players...")
    colnames = Set(Symbol.(names(df)))

    ovr_col = if :overall_rating in colnames
        :overall_rating
    elseif :ovr in colnames
        :ovr
    else
        error("No OVR column found. Expected one of: :overall_rating or :ovr")
    end

    wage_col = if :wage in colnames
        :wage
    else
        error("No wage column found. Expected :wage")
    end

    # Filter players with valid wages (>0) and reasonable OVR (>=70)
    valid_mask = (coalesce.(df[!, wage_col], 0.0) .> 0) .& (coalesce.(df[!, ovr_col], 0) .>= 70)
    valid_players = df[valid_mask, :]

    # Sort by OVR to get top players
    sort!(valid_players, ovr_col, rev=true)

    # Take top 2000 or all available if less
    sample_size = min(2000, nrow(valid_players))
    top_players = first(valid_players, sample_size)

    println("   Using $(sample_size) top players for wage inference")

    # Calculate average wage by exact OVR
    wage_by_ovr = Dict{Int, Float64}()

    for ovr in 70:99
        ovr_players = top_players[top_players[!, ovr_col] .== ovr, :]
        if nrow(ovr_players) > 0
            wage_by_ovr[ovr] = mean(skipmissing(Float64.(ovr_players[!, wage_col])))
        end
    end

    # Fill gaps using linear interpolation
    # For OVRs without data, interpolate from nearest available values
    filled_wage_by_ovr = Dict{Int, Float64}()

    for ovr in 70:99
        if haskey(wage_by_ovr, ovr)
            filled_wage_by_ovr[ovr] = wage_by_ovr[ovr]
        else
            # Find nearest lower and upper OVRs with data
            lower_ovr = nothing
            upper_ovr = nothing

            for delta in 1:10
                if haskey(wage_by_ovr, ovr - delta)
                    lower_ovr = ovr - delta
                    break
                end
            end

            for delta in 1:10
                if haskey(wage_by_ovr, ovr + delta)
                    upper_ovr = ovr + delta
                    break
                end
            end

            # Interpolate or extrapolate
            if !isnothing(lower_ovr) && !isnothing(upper_ovr)
                # Linear interpolation
                ratio = (ovr - lower_ovr) / (upper_ovr - lower_ovr)
                filled_wage_by_ovr[ovr] = wage_by_ovr[lower_ovr] + ratio * (wage_by_ovr[upper_ovr] - wage_by_ovr[lower_ovr])
            elseif !isnothing(lower_ovr)
                # Extrapolate from lower bound
                filled_wage_by_ovr[ovr] = wage_by_ovr[lower_ovr] * (1.05 ^ (ovr - lower_ovr))
            elseif !isnothing(upper_ovr)
                # Extrapolate from upper bound
                filled_wage_by_ovr[ovr] = wage_by_ovr[upper_ovr] * (0.95 ^ (upper_ovr - ovr))
            else
                # Fallback: use simple formula (OVR × 2000)
                filled_wage_by_ovr[ovr] = Float64(ovr * 2000)
            end
        end
    end

    # Add fallback for OVRs below 70 (young/low quality free agents)
    for ovr in 40:69
        filled_wage_by_ovr[ovr] = Float64(ovr * 1000)  # Simple linear
    end

    println("   ✅ Wage inference complete for OVR 40-99")
    println("   Sample wages: OVR 70 = €$(round(Int, filled_wage_by_ovr[70])), OVR 80 = €$(round(Int, filled_wage_by_ovr[80])), OVR 90 = €$(round(Int, filled_wage_by_ovr[90]))")

    return filled_wage_by_ovr
end

"""
Returns the signing bonus multiplier for free agents based on their OVR.
Higher quality free agents demand significantly higher signing bonuses.

OVR brackets:
- 70-74: 3.0x
- 75-79: 4.0x
- 80-84: 5.0x
- 85+:   7.0x
"""
function get_free_agent_signing_multiplier(ovr::Int)
    if ovr >= 85
        return 7.0
    elseif ovr >= 80
        return 5.0
    elseif ovr >= 75
        return 4.0
    elseif ovr >= 70
        return 3.0
    else
        return 2.0  # Low quality free agents
    end
end