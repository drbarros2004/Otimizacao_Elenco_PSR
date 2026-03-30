using CSV, DataFrames, Dates, TOML

include("utils.jl")

# --- PATH CONFIGURATIONS ---
const RAW_BASE_PATH = "data/raw/player_stats.csv"
const RAW_CUSTOM_PATH = "data/raw/stats_brasileirao_with_correct_teams.csv"
const PROCESSED_OUTPUT_PATH = "data/processed/processed_player_data.csv"
const PLAYER_WINDOW_AUDIT_PATH = "data/processed/player_window_audit.csv"
const MARKET_CONFIG_PATH = "config/market_settings.toml"

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
        return occursin("brazil", league_norm) ||
               occursin("brasileir", league_norm) ||
               occursin("série a", league_norm) ||
               occursin("serie a", league_norm)
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
