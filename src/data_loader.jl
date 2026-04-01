using CSV, DataFrames, Dates, Random, TOML

include("utils.jl")

# --- PATH CONFIGURATIONS ---
const RAW_BASE_PATH = "data/raw/player_stats.csv"
const RAW_CUSTOM_PATH = "data/raw/stats_brasileirao_with_correct_teams.csv"
const PROCESSED_OUTPUT_PATH = "data/processed/processed_player_data.csv"
const PLAYER_WINDOW_AUDIT_PATH = "data/processed/player_window_audit.csv"
const PLAYER_NODE_AUDIT_PATH = "data/processed/player_node_audit.csv"
const MARKET_CONFIG_PATH = "config/market_settings.toml"
const DEFAULT_PANTUSO_SIGMA = 0.1791

# Column mapping from the SoFIFA Scraper
const SCRAPER_COLUMNS = [
    :player_id, :name, :dob, :positions, :overall_rating, 
    :potential, :value, :wage, :international_reputation,
    :club_name, :club_league_name
]

"""
Loads raw player data from base and custom sources, merges them, 
handles duplicates, and performs initial feature engineering (age calculation).
"""
function load_and_clean_data()
    println("Starting data ingestion using SoFIFA schema...")
    
    # Load raw CSV files
    if !isfile(RAW_BASE_PATH) || !isfile(RAW_CUSTOM_PATH)
        error("Source files not found in data/raw/. Check your paths.")
    end

    df_base = CSV.read(RAW_BASE_PATH, DataFrame)
    df_custom = CSV.read(RAW_CUSTOM_PATH, DataFrame)

    # Remove Brazilian-team rows from base source to avoid unlicensed players
    # Keep Brazilian teams only from the custom source
    custom_clubs = Set(
        lowercase.(strip.(collect(skipmissing(df_custom.club_name))))
    )
    
    is_brazilian_league(league) = begin
        if ismissing(league)
            return false
        end
        league_norm = lowercase(strip(String(league)))
        return occursin("serie a", league_norm)
    end

    base_keep_mask = map(eachrow(df_base)) do row
        club_norm = ismissing(row.club_name) ? "" : lowercase(strip(String(row.club_name)))
        remove_by_club = !isempty(club_norm) && (club_norm in custom_clubs)
        remove_by_league = is_brazilian_league(row.club_league_name)
        return !(remove_by_club || remove_by_league)
    end

    base_before = nrow(df_base)
    df_base = df_base[base_keep_mask, :]
    println("🧹 Removed $(base_before - nrow(df_base)) Brazilian-team rows from base dataset.")

    # The custom data doesn't include the league, so we have to hard-code it in here
    df_custom.club_league_name .= "Brazil Serie A"
    
    # Merge datasets via vertical concatenation
    df_total = vcat(df_base, df_custom, cols=:union)
    
    # Remove duplicates by player_id 
    n_before = nrow(df_total)
    unique!(df_total, :player_id)
    println("✂️ Removed $(n_before - nrow(df_total)) duplicate records.")
    
    # Filters only relevant columns
    df_cleaned = df_total[:, intersect(names(df_total), String.(SCRAPER_COLUMNS))]
    
    # Calculating the age of the players
    df_cleaned.age = calculate_age.(df_cleaned.dob)

    # Map positions to granular tactical groups (e.g., GK, CB, CM, ST)
    df_cleaned.pos_group = map_position_group.(df_cleaned.positions)

    # Null Handling (Imputation)
    df_cleaned.value = Float64.(coalesce.(df_cleaned.value, 0.0))
    df_cleaned.international_reputation = Float64.(coalesce.(df_cleaned.international_reputation, 1.0))
    df_cleaned.club_name = coalesce.(df_cleaned.club_name, "Unknown")
    
    # Persistence
    mkpath(dirname(PROCESSED_OUTPUT_PATH))
    CSV.write(PROCESSED_OUTPUT_PATH, df_cleaned)
    
    println("Success! Total unique players: $(nrow(df_cleaned))")
    println("Generated 'age' column for future growth/decay projections.")
    
    return df_cleaned
end


"""
    filter_top_k_players_by_position(df::DataFrame, k::Int; protected_player_ids::Vector{Int}=Int[])

Keeps only the top-K players per position group based on overall quality proxies,
optionally preserving a set of protected players (for example, players from the
initial squad strategy team).
"""
function filter_top_k_players_by_position(
    df::DataFrame,
    k::Int;
    protected_player_ids::Vector{Int} = Int[]
)
    if k <= 0
        return df
    end

    required_cols = [:player_id, :pos_group, :overall_rating, :potential, :value]
    for col in required_cols
        if !hasproperty(df, col)
            error("Cannot apply top-K filter: missing required column '$col'.")
        end
    end

    sorted_df = sort(df, [:pos_group, :overall_rating, :potential, :value], rev=[false, true, true, true])
    grouped = groupby(sorted_df, :pos_group)

    kept_parts = DataFrame[]
    for g in grouped
        n_keep = min(k, nrow(g))
        push!(kept_parts, first(g, n_keep))
    end

    filtered = isempty(kept_parts) ? DataFrame() : vcat(kept_parts..., cols=:union)

    if !isempty(protected_player_ids)
        protected_set = Set(Int.(protected_player_ids))
        already_kept = Set(Int.(filtered.player_id))
        missing_ids = collect(setdiff(protected_set, already_kept))

        if !isempty(missing_ids)
            extra = df[in.(Int.(df.player_id), Ref(Set(missing_ids))), :]
            filtered = vcat(filtered, extra, cols=:union)
        end
    end

    unique!(filtered, :player_id)
    return filtered
end


"""
Simulates player evolution across multiple windows sequentially.
Returns two maps:
1. `ovr_map`: (player_id, window) -> OVR
2. `value_map`: (player_id, window) -> Value
"""
function generate_projections(df_players::DataFrame, num_windows::Int)
    println("Simulating multi-period evolution (OVR, Value & Growth Potential)...")

    ovr_map = Dict{Tuple{Int, Int}, Int}()
    value_map = Dict{Tuple{Int, Int}, Float64}()
    growth_potential_map = Dict{Tuple{Int, Int}, Float64}()

    for row in eachrow(df_players)
        p_id = row.player_id

        # Initial State (Window 0)
        current_ovr = Float64(row.overall_rating)
        current_val = Float64(row.value)
        current_age = Int(row.age)
        potential = Float64(row.potential)

        ovr_map[(p_id, 0)] = Int(current_ovr)
        value_map[(p_id, 0)] = current_val

        # Calculate growth potential for window 0 using evolution_step logic
        gap = max(0.0, potential - current_ovr)
        growth_coeff = max(0.02, 0.28 - (0.01 * current_age))
        growth_potential_map[(p_id, 0)] = round(gap * growth_coeff, digits=2)

        # Simulate subsequent windows
        for w in 1:num_windows

            new_ovr, new_val, new_age, status = evolution_step(
                current_ovr,
                potential,
                current_age,
                current_val,
                w
            )

            # Update state for next window iteration
            current_ovr = Float64(new_ovr)
            current_val = new_val
            current_age = new_age

            # Store in maps
            ovr_map[(p_id, w)] = new_ovr
            value_map[(p_id, w)] = round(new_val, digits=2)

            # Calculate growth potential using same logic as evolution_step
            gap = max(0.0, potential - Float64(new_ovr))
            growth_coeff = max(0.02, 0.28 - (0.01 * new_age))
            growth_potential_map[(p_id, w)] = round(gap * growth_coeff, digits=2)
        end
    end

    println("Projections completed for $(nrow(df_players)) players.")
    println("   ✅ OVR map: $(length(ovr_map)) entries")
    println("   ✅ Value map: $(length(value_map)) entries")
    println("   ✅ Growth Potential map: $(length(growth_potential_map)) entries")

    return ovr_map, value_map, growth_potential_map
end

"""
Builds tactical requirements indexed by node, following N_{r,n} notation.
Returns a dictionary with key (position_group, node_id) -> required_count.
"""
function generate_node_position_requirements(tree::ScenarioTree)
    req_map = Dict{Tuple{String, Int}, Int}()

    for (node_id, node) in tree.nodes
        for (pos_group, required_count) in node.position_requirements
            req_map[(pos_group, node_id)] = required_count
        end
    end

    return req_map
end

function _resolve_stochastic_sigmas(stochastic_cfg::StochasticConfig)
    ovr_sigma = stochastic_cfg.ovr_shock_sigma > 0 ? stochastic_cfg.ovr_shock_sigma : DEFAULT_PANTUSO_SIGMA
    value_sigma = stochastic_cfg.value_shock_sigma > 0 ? stochastic_cfg.value_shock_sigma : DEFAULT_PANTUSO_SIGMA
    return ovr_sigma, value_sigma
end

function _normalize_player_name(name)::String
    return lowercase(strip(replace(String(name), r"\s+" => " ")))
end

function _build_player_name_index(df_players::DataFrame)
    index = Dict{String, Vector{Int}}()
    for row in eachrow(df_players)
        name = _normalize_player_name(coalesce(row.name, ""))
        if isempty(name)
            continue
        end
        ids = get!(index, name, Int[])
        push!(ids, Int(row.player_id))
    end
    return index
end

function _resolve_manual_injury_ids(
    raw_overrides,
    player_name_index::Dict{String, Vector{Int}}
)
    if !(raw_overrides isa AbstractVector)
        error("manual_events.injury_overrides must be an array of names or player_ids.")
    end

    injury_ids = Set{Int}()
    unresolved = String[]
    ambiguous = String[]

    for item in raw_overrides
        if item isa Integer
            push!(injury_ids, Int(item))
            continue
        end

        name = _normalize_player_name(String(item))
        if isempty(name)
            continue
        end

        matched_ids = get(player_name_index, name, Int[])
        if isempty(matched_ids)
            push!(unresolved, String(item))
            continue
        end

        if length(matched_ids) > 1
            push!(ambiguous, String(item))
        end

        for p_id in matched_ids
            push!(injury_ids, p_id)
        end
    end

    return injury_ids, unresolved, ambiguous
end

function _apply_manual_event_to_node!(
    node::ScenarioNode,
    node_id::Int,
    stochastic_cfg::StochasticConfig,
    player_name_index::Dict{String, Vector{Int}}
)
    event = get(stochastic_cfg.manual_node_events, node_id, nothing)
    if isnothing(event)
        return Dict{String, Any}()
    end

    event_state = Dict{String, Any}()

    label = get(event, "label", "")
    if !isempty(strip(String(label)))
        node.metadata["manual_label"] = String(label)
    end

    if haskey(event, "tactical_override")
        scheme = String(event["tactical_override"])
        node.metadata["tactical_override"] = scheme
        event_state["tactical_override"] = scheme

        if haskey(event, "position_requirements")
            empty!(node.position_requirements)
            for (pos, count) in event["position_requirements"]
                node.position_requirements[String(pos)] = Int(count)
            end
        end
    end

    if haskey(event, "market_shock")
        event_state["market_shock"] = Float64(event["market_shock"])
        node.metadata["manual_market_shock"] = Float64(event["market_shock"])
    end

    if haskey(event, "chemistry_bonus")
        event_state["chemistry_bonus"] = Float64(event["chemistry_bonus"])
        node.metadata["manual_chemistry_bonus"] = Float64(event["chemistry_bonus"])
    end

    if haskey(event, "injury_overrides")
        injury_ids, unresolved, ambiguous = _resolve_manual_injury_ids(event["injury_overrides"], player_name_index)
        event_state["forced_injury_ids"] = injury_ids
        node.metadata["manual_injury_ids"] = sort(collect(injury_ids))

        if !isempty(unresolved)
            @warn "Unresolved manual injury overrides at node $node_id: $(join(unresolved, ", "))"
        end
        if !isempty(ambiguous)
            @warn "Ambiguous manual injury override names at node $node_id (mapped to multiple ids): $(join(ambiguous, ", "))"
        end
    end

    return event_state
end

"""
Simulates stochastic evolution over a scenario tree using node indexing.

Returns maps with keys (player_id, node_id):
1. ovr_node_map
2. value_node_map
3. growth_potential_node_map
4. starter_allowed_map  (1 if can start, 0 if unavailable/injured)
5. sell_allowed_map     (1 if sale allowed, 0 if sale blocked)
6. chemistry_multiplier_map (node_id -> multiplier, can be 1.0, 0.0, or negative)
"""
function generate_stochastic_projections(
    df_players::DataFrame,
    tree::ScenarioTree,
    stochastic_cfg::StochasticConfig;
    rng_seed::Int = 42
)
    println("Simulating stochastic node-based evolution (OVR, Value, Availability & Chemistry)...")

    rng = MersenneTwister(rng_seed)
    ovr_sigma, value_sigma = _resolve_stochastic_sigmas(stochastic_cfg)

    ovr_node_map = Dict{Tuple{Int, Int}, Int}()
    value_node_map = Dict{Tuple{Int, Int}, Float64}()
    growth_potential_node_map = Dict{Tuple{Int, Int}, Float64}()
    age_node_map = Dict{Tuple{Int, Int}, Int}()
    starter_allowed_map = Dict{Tuple{Int, Int}, Int}()
    sell_allowed_map = Dict{Tuple{Int, Int}, Int}()
    chemistry_multiplier_map = Dict{Int, Float64}()
    player_name_index = _build_player_name_index(df_players)

    # Cache immutable player attributes for performance and stable lookups.
    player_ids = Int.(df_players.player_id)
    potential_by_player = Dict{Int, Float64}(Int(row.player_id) => Float64(coalesce(row.potential, row.overall_rating)) for row in eachrow(df_players))
    base_ovr_by_player = Dict{Int, Float64}(Int(row.player_id) => Float64(coalesce(row.overall_rating, 60.0)) for row in eachrow(df_players))
    base_value_by_player = Dict{Int, Float64}(Int(row.player_id) => Float64(coalesce(row.value, 0.5)) for row in eachrow(df_players))
    base_age_by_player = Dict{Int, Int}(Int(row.player_id) => Int(coalesce(row.age, 25)) for row in eachrow(df_players))

    root_id = tree.root_id

    # Root node is deterministic initial state (no anticipation needed).
    root_node = tree.nodes[root_id]
    root_event_state = _apply_manual_event_to_node!(root_node, root_id, stochastic_cfg, player_name_index)
    root_effective_scheme = haskey(root_event_state, "tactical_override") ? String(root_event_state["tactical_override"]) : String(root_node.tactical_scheme)
    root_node.metadata["effective_tactical_scheme"] = root_effective_scheme

    root_chemistry_multiplier = haskey(root_event_state, "chemistry_bonus") ? Float64(root_event_state["chemistry_bonus"]) : 1.0
    chemistry_multiplier_map[root_id] = root_chemistry_multiplier
    root_node.metadata["chemistry_multiplier"] = root_chemistry_multiplier

    root_injury_ids = haskey(root_event_state, "forced_injury_ids") ? root_event_state["forced_injury_ids"] : Set{Int}()

    for p_id in player_ids
        root_ovr = round(Int, base_ovr_by_player[p_id])
        root_value = max(0.5, base_value_by_player[p_id])
        root_age = base_age_by_player[p_id]

        ovr_node_map[(p_id, root_id)] = root_ovr
        value_node_map[(p_id, root_id)] = round(root_value, digits=2)
        age_node_map[(p_id, root_id)] = root_age

        gap = max(0.0, potential_by_player[p_id] - Float64(root_ovr))
        growth_coeff = max(0.02, 0.28 - (0.01 * root_age))
        growth_potential_node_map[(p_id, root_id)] = round(gap * growth_coeff, digits=2)

        injured_root = p_id in root_injury_ids
        starter_allowed_map[(p_id, root_id)] = injured_root ? 0 : 1
        sell_allowed_map[(p_id, root_id)] = injured_root ? 0 : 1
        root_node.sell_allowed[p_id] = !injured_root
        if injured_root
            push!(root_node.injury_players, p_id)
        end
    end
    root_node.metadata["injury_count"] = length(root_node.injury_players)

    stage_ids = sort(collect(keys(tree.nodes_by_stage)))

    for stage in stage_ids
        if stage == 0
            continue
        end

        node_ids = get(tree.nodes_by_stage, stage, Int[])
        for node_id in node_ids
            node = tree.nodes[node_id]
            parent_id = something(node.parent_id, root_id)
            parent_node = tree.nodes[parent_id]
            parent_effective_scheme = get_effective_tactical_scheme(parent_node)
            event_state = _apply_manual_event_to_node!(node, node_id, stochastic_cfg, player_name_index)
            forced_injury_ids = haskey(event_state, "forced_injury_ids") ? event_state["forced_injury_ids"] : Set{Int}()

            if haskey(event_state, "tactical_override")
                node.metadata["effective_tactical_scheme"] = String(event_state["tactical_override"])
            else
                # Inherit tactical setup from parent when no local override is declared.
                node.metadata["effective_tactical_scheme"] = parent_effective_scheme
                empty!(node.position_requirements)
                for (pos, count) in parent_node.position_requirements
                    node.position_requirements[pos] = count
                end
            end

            conflict_occurs = rand(rng) < stochastic_cfg.chemistry_conflict_probability
            chemistry_multiplier = if conflict_occurs
                rand(rng) < 0.5 ? 0.0 : -0.5
            else
                1.0
            end
            if haskey(event_state, "chemistry_bonus")
                chemistry_multiplier = Float64(event_state["chemistry_bonus"])
            end
            chemistry_multiplier_map[node_id] = chemistry_multiplier
            node.metadata["chemistry_multiplier"] = chemistry_multiplier

            # Node-level market noise creates coherent branch movement across players.
            branch_market_shock = randn(rng) * value_sigma * 0.15
            if haskey(event_state, "market_shock")
                branch_market_shock += Float64(event_state["market_shock"])
            end
            node.metadata["branch_market_shock"] = branch_market_shock

            for p_id in player_ids
                prev_ovr = Float64(ovr_node_map[(p_id, parent_id)])
                prev_val = value_node_map[(p_id, parent_id)]
                prev_age = age_node_map[(p_id, parent_id)]
                potential = potential_by_player[p_id]

                det_ovr, det_val, det_age, _ = evolution_step(
                    prev_ovr,
                    potential,
                    prev_age,
                    prev_val,
                    stage
                )

                ovr_shock = randn(rng) * ovr_sigma
                value_shock = randn(rng) * value_sigma

                shocked_ovr = clamp(round(Int, det_ovr + ovr_shock), 40, 99)
                value_factor = max(0.4, 1.0 + value_shock + branch_market_shock)
                shocked_value = max(0.5, det_val * value_factor)

                ovr_node_map[(p_id, node_id)] = shocked_ovr
                value_node_map[(p_id, node_id)] = round(shocked_value, digits=2)
                age_node_map[(p_id, node_id)] = det_age

                gap = max(0.0, potential - Float64(shocked_ovr))
                growth_coeff = max(0.02, 0.28 - (0.01 * det_age))
                growth_potential_node_map[(p_id, node_id)] = round(gap * growth_coeff, digits=2)

                injured_now = (rand(rng) < stochastic_cfg.injury_probability) || (p_id in forced_injury_ids)
                starter_allowed_map[(p_id, node_id)] = injured_now ? 0 : 1
                sell_allowed_map[(p_id, node_id)] = injured_now ? 0 : 1

                if injured_now
                    push!(node.injury_players, p_id)
                end
                node.sell_allowed[p_id] = !injured_now
            end

            node.metadata["injury_count"] = length(node.injury_players)
        end
    end

    println("Stochastic projections completed for $(nrow(df_players)) players across $(length(tree.nodes)) nodes.")
    println("   ✅ OVR node map: $(length(ovr_node_map)) entries")
    println("   ✅ Value node map: $(length(value_node_map)) entries")
    println("   ✅ Growth potential node map: $(length(growth_potential_node_map)) entries")

    return (
        ovr_node_map,
        value_node_map,
        growth_potential_node_map,
        starter_allowed_map,
        sell_allowed_map,
        chemistry_multiplier_map
    )
end

"""
Maps SoFIFA position strings to granular tactical position groups.
Returns the player's primary position group.

# Arguments
- `position_string::AbstractString`: Raw position from SoFIFA (e.g., "ST, LW, LM" or "CB")

# Returns
- `String`: One of "GK", "CB", "CM", "RB", "LB", "RW", "LW", "ST"
"""
function map_position_group(position_string::AbstractString)
    # Extract first position (primary position)
    primary_pos = strip(split(String(position_string), ",")[1])

    # Position mapping
    if primary_pos == "GK"
        return "GK"
    elseif primary_pos in ["RB", "RWB"]
        return "RB"
    elseif primary_pos in ["LB", "LWB"]
        return "LB"
    elseif primary_pos in ["CB"]
        return "CB"
    elseif primary_pos in ["CDM", "CM", "CAM"]
        return "CM"
    elseif primary_pos in ["RM", "RW"]
        return "RW"
    elseif primary_pos in ["LW", "LM"]
        return "LW"
    elseif primary_pos in ["ST", "CF"]
        return "ST"
    else
        @warn "Unknown position: $primary_pos. Defaulting to CM."
        return "CM"
    end
end

"""
Generates a map of wages (annual salary in EUR) for all players across windows.
For simplicity, wage is assumed constant unless the player significantly improves.

For free agents (players with wage = 0 or very low), this function infers
realistic wages based on the OVR distribution of top players in the dataset.
"""
function generate_wage_map(df_players::DataFrame, ovr_map::Dict, num_windows::Int)
    println("Generating Wage map...")

    # Load optional market settings for free-agent wage inflation
    free_agent_wage_threshold = 1000.0
    free_agent_wage_multiplier = 1.0

    if isfile(MARKET_CONFIG_PATH)
        market_cfg = TOML.parsefile(MARKET_CONFIG_PATH)
        market_settings = get(market_cfg, "market_settings", Dict{String,Any}())

        free_agent_wage_threshold = Float64(get(market_settings, "free_agent_wage_threshold", free_agent_wage_threshold))
        free_agent_wage_multiplier = Float64(get(market_settings, "free_agent_wage_multiplier", free_agent_wage_multiplier))

        if free_agent_wage_multiplier <= 0
            @warn "Invalid free_agent_wage_multiplier=$free_agent_wage_multiplier. Falling back to 1.0"
            free_agent_wage_multiplier = 1.0
        end
    end

    println("   Free-agent wage threshold: €$(round(Int, free_agent_wage_threshold))")
    println("   Free-agent wage multiplier: x$(round(free_agent_wage_multiplier, digits=2))")

    # Infer realistic wages for free agents based on OVR distribution
    wage_by_ovr = infer_free_agent_wages(df_players)

    wage_map = Dict{Tuple{Int, Int}, Float64}()

    for row in eachrow(df_players)
        p_id = row.player_id
        base_ovr = row.overall_rating
        raw_wage = Float64(coalesce(row.wage, 0.0))
        is_free_agent = raw_wage <= free_agent_wage_threshold

        # Determine base wage:
        # If player has valid wage (>1000), use it
        # If player is a free agent (wage ≤ 1000), infer from OVR
        base_wage = if !is_free_agent
            raw_wage
        else
            # Free agent - inferred wage with configurable inflation multiplier
            inferred = get(wage_by_ovr, Int(round(base_ovr)), base_ovr * 2000)
            inferred * free_agent_wage_multiplier
        end

        for w in 0:num_windows
            current_ovr = ovr_map[(p_id, w)]

            # Wage adjusts with OVR improvement (simple model)
            ovr_factor = current_ovr / max(base_ovr, 1)
            adjusted_wage = base_wage * ovr_factor

            wage_map[(p_id, w)] = round(adjusted_wage, digits=2)
        end
    end

    println("✅ Wage map generated (with inferred wages for free agents).")
    return wage_map
end

"""
Generates annual wages indexed by node: (player_id, node_id) -> wage.
"""
function generate_wage_map_by_node(df_players::DataFrame, ovr_node_map::Dict, tree::ScenarioTree)
    println("Generating node-indexed Wage map...")

    free_agent_wage_threshold = 1000.0
    free_agent_wage_multiplier = 1.0

    if isfile(MARKET_CONFIG_PATH)
        market_cfg = TOML.parsefile(MARKET_CONFIG_PATH)
        market_settings = get(market_cfg, "market_settings", Dict{String,Any}())

        free_agent_wage_threshold = Float64(get(market_settings, "free_agent_wage_threshold", free_agent_wage_threshold))
        free_agent_wage_multiplier = Float64(get(market_settings, "free_agent_wage_multiplier", free_agent_wage_multiplier))

        if free_agent_wage_multiplier <= 0
            @warn "Invalid free_agent_wage_multiplier=$free_agent_wage_multiplier. Falling back to 1.0"
            free_agent_wage_multiplier = 1.0
        end
    end

    wage_by_ovr = infer_free_agent_wages(df_players)
    wage_node_map = Dict{Tuple{Int, Int}, Float64}()

    node_ids = sort(collect(keys(tree.nodes)))

    for row in eachrow(df_players)
        p_id = Int(row.player_id)
        base_ovr = Float64(coalesce(row.overall_rating, 60.0))
        raw_wage = Float64(coalesce(row.wage, 0.0))
        is_free_agent = raw_wage <= free_agent_wage_threshold

        base_wage = if !is_free_agent
            raw_wage
        else
            inferred = get(wage_by_ovr, Int(round(base_ovr)), base_ovr * 2000)
            inferred * free_agent_wage_multiplier
        end

        for node_id in node_ids
            current_ovr = Float64(ovr_node_map[(p_id, node_id)])
            ovr_factor = current_ovr / max(base_ovr, 1.0)
            adjusted_wage = base_wage * ovr_factor
            wage_node_map[(p_id, node_id)] = round(adjusted_wage, digits=2)
        end
    end

    println("✅ Node-indexed wage map generated.")
    return wage_node_map
end

"""
Calculates the actual Acquisition Cost (Buying Price) using the Market Reputation logic.
This map is what the Gurobi budget constraint will use.
"""
function generate_cost_map(df_players::DataFrame, value_map::Dict, num_windows::Int)
    println("Applying Market Reputation to acquisition costs...")
    
    # Load configuration
    if !isfile(MARKET_CONFIG_PATH)
        @warn "Market config not found at $MARKET_CONFIG_PATH. Using baseline costs."
        return value_map # Fallback to 1:1 value if config is missing
    end
    market_cfg = TOML.parsefile(MARKET_CONFIG_PATH)
    
    cost_map = Dict{Tuple{Int, Int}, Float64}()

    for row in eachrow(df_players)
        p_id = row.player_id
        league = coalesce(row.club_league_name, "Unknown")
        ir = Int(row.international_reputation)
        
        # Calculate the fixed multiplier for this specific player/league combo
        multiplier = get_market_multiplier(league, ir, market_cfg)
        
        for w in 0:num_windows
            # Cost = Market Value at window W * Reputation Multiplier
            base_value = value_map[(p_id, w)]
            cost_map[(p_id, w)] = round(base_value * multiplier, digits=2)
        end
    end
    
    println("✅ Cost map generated for all windows.")
    return cost_map
end

"""
Generates acquisition costs indexed by node: (player_id, node_id) -> cost.
Uses the same reputation logic as deterministic mode on top of node values.
"""
function generate_cost_map_by_node(df_players::DataFrame, value_node_map::Dict, tree::ScenarioTree)
    println("Applying Market Reputation to node-indexed acquisition costs...")

    if !isfile(MARKET_CONFIG_PATH)
        @warn "Market config not found at $MARKET_CONFIG_PATH. Using node value map as fallback costs."
        return value_node_map
    end

    market_cfg = TOML.parsefile(MARKET_CONFIG_PATH)
    cost_node_map = Dict{Tuple{Int, Int}, Float64}()
    node_ids = sort(collect(keys(tree.nodes)))

    for row in eachrow(df_players)
        p_id = Int(row.player_id)
        league = coalesce(row.club_league_name, "Unknown")
        ir = Int(coalesce(row.international_reputation, 1.0))

        multiplier = get_market_multiplier(league, ir, market_cfg)

        for node_id in node_ids
            base_value = value_node_map[(p_id, node_id)]
            cost_node_map[(p_id, node_id)] = round(base_value * multiplier, digits=2)
        end
    end

    println("✅ Node-indexed cost map generated.")
    return cost_node_map
end

"""
Exports a node-level analytical report for stochastic projections.
"""
function export_node_analysis(
    df_players::DataFrame,
    tree::ScenarioTree,
    ovr_node_map::Dict,
    value_node_map::Dict,
    cost_node_map::Dict,
    starter_allowed_map::Dict,
    sell_allowed_map::Dict
)
    println("📊 Generating node-level audit report...")

    rows = []
    sorted_nodes = sort(collect(tree.nodes); by=x -> x[1])

    for row in eachrow(df_players)
        p_id = Int(row.player_id)

        for (node_id, node) in sorted_nodes
            effective_scheme = get_effective_tactical_scheme(node)
            push!(rows, (
                player_id = p_id,
                name = row.name,
                node_id = node_id,
                stage = node.stage,
                parent_id = isnothing(node.parent_id) ? -1 : node.parent_id,
                branch_probability = node.branch_probability,
                cumulative_probability = node.cumulative_probability,
                tactical_scheme = effective_scheme,
                scenario_label = get(node.metadata, "manual_label", missing),
                chemistry_multiplier = Float64(get(node.metadata, "chemistry_multiplier", 1.0)),
                ovr = ovr_node_map[(p_id, node_id)],
                market_value_eur = round(value_node_map[(p_id, node_id)], digits=0),
                acquisition_cost_eur = round(cost_node_map[(p_id, node_id)], digits=0),
                starter_allowed = starter_allowed_map[(p_id, node_id)],
                sell_allowed = sell_allowed_map[(p_id, node_id)]
            ))
        end
    end

    df_node = DataFrame(rows)
    CSV.write(PLAYER_NODE_AUDIT_PATH, df_node)

    println("✅ Player-node audit saved at '$PLAYER_NODE_AUDIT_PATH'.")
    return df_node
end

"""
    export_analysis(df_players::DataFrame, ovr_map::Dict, value_map::Dict, cost_map::Dict, num_windows::Int)

Creates a unified player-window audit report with separate columns for League Multiplier 
and IR Bonus for detailed financial transparency.
"""
function export_analysis(df_players::DataFrame, ovr_map::Dict, value_map::Dict, cost_map::Dict, num_windows::Int)
    println("📊 Generating Master Audit Report with Financial Breakdown...")
    
    # Carregar config para extrair os componentes do multiplicador
    market_cfg = TOML.parsefile("config/market_settings.toml")
    buyer_rep = market_cfg["buying_club"]["reputation"]
    default_rep = market_cfg["market_settings"]["default_league_reputation"]
    ir_weight = market_cfg["market_settings"]["ir_multiplier"]
    leagues_dict = market_cfg["leagues"]

    analysis_rows = []

    for row in eachrow(df_players)
        p_id = row.player_id
        league = coalesce(row.club_league_name, "Unknown")
        ir = Int(row.international_reputation)
        start_age = Int(row.age)
        
        # --- Cálculo dos Componentes ---
        origin_rep = get(leagues_dict, league, default_rep)
        # Multiplicador da Liga (Base 1.0)
        l_mult = round(1.0 + max(0.0, (origin_rep - buyer_rep) / buyer_rep), digits=2)
        # Bônus de IR (Ex: 0.1 por estrela acima de 1)
        i_bonus = round((ir - 1) * ir_weight, digits=2)

        for w in 0:num_windows
            val = value_map[(p_id, w)]
            cost = cost_map[(p_id, w)]
            
            # No seu src/data_loader.jl, dentro do push! da função export_master_analysis:

        push!(analysis_rows, (
            player_id = p_id,
            name = row.name,
            window = w,
            age = start_age + floor(Int, w / 2),
            ovr = ovr_map[(p_id, w)],
            league = league,
            ir_stars = ir,
            league_mult = l_mult,
            ir_bonus = i_bonus,
            rep_multiplier = val > 0 ? round(cost / val, digits=2) : 1.0, # <--- MUDADO PARA rep_multiplier
            market_value_eur = round(val, digits=0),
            acquisition_cost_eur = round(cost, digits=0)
        ))
        end
    end

    df_master = DataFrame(analysis_rows)
    CSV.write(PLAYER_WINDOW_AUDIT_PATH, df_master)

    # Remove legacy artifacts now covered by the unified player-window audit.
    legacy_paths = [
        "data/processed/master_audit.csv",
        "data/processed/market_analysis.csv",
        "data/processed/evolution_analysis.csv"
    ]
    for path in legacy_paths
        if isfile(path)
            rm(path; force=true)
        end
    end

    println("✅ Player-window audit saved with financial breakdown.")
    return df_master
end
