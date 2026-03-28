using CSV, DataFrames, Dates, TOML

include("utils.jl")

# --- PATH CONFIGURATIONS ---
const RAW_BASE_PATH = "data/raw/player_stats.csv"
const RAW_CUSTOM_PATH = "data/raw/stats_brasileirao_with_correct_teams.csv"
const PROCESSED_OUTPUT_PATH = "data/processed/processed_player_data.csv"
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
    
    # Null Handling (Imputation)
    df_cleaned.value = Float64.(coalesce.(df_cleaned.value, 0.0))
    df_cleaned.international_reputation = Float64.(coalesce.(df_cleaned.international_reputation, 1.0))
    
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
1. ovr_map: (player_id, window) -> OVR
2. value_map: (player_id, window) -> Value
"""
function generate_projections(df_players::DataFrame, num_windows::Int)
    println("Simulating multi-period evolution (OVR & Value)...")
    
    ovr_map = Dict{Tuple{Int, Int}, Int}()
    value_map = Dict{Tuple{Int, Int}, Float64}()
    
    # Set seed for reproducibility 
    Random.seed!(42)

    for row in eachrow(df_players)
        p_id = row.player_id
        
        # Initial State (Window 0)
        current_ovr = Float64(row.overall_rating)
        current_val = Float64(row.value)
        current_age = Int(row.age)
        
        ovr_map[(p_id, 0)] = Int(current_ovr)
        value_map[(p_id, 0)] = current_val
        
        # Simulate subsequent windows
        for w in 1:num_windows

            new_ovr, new_val, new_age = evolution_step(
                current_ovr, 
                Float64(row.potential), 
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
        end
    end
    
    println("Projections completed for $(nrow(df_players)) players.")
    return ovr_map, value_map
end

"""
Sequential simulation: Window(T) depends on Window(T-1).
"""
function export_evolution_analysis(df_players::DataFrame, num_windows::Int)
    println("Running sequential simulation for $num_windows windows...")
    Random.seed!(42) # For reproducibility
    
    analysis_rows = []

    for row in eachrow(df_players)

        # Initial State (Window 0)
        curr_ovr = Float64(row.overall_rating)
        curr_val = Float64(row.value)
        curr_age = Int(row.age)
        curr_pot = Float64(row.potential)
        
        # Save Window 0
        push!(analysis_rows, (
            player_id = row.player_id, name = row.name, 
            window = 0, age = curr_age, 
            ovr = round(curr_ovr, digits=1), value = round(curr_val, digits=2),
            status = "Initial"
        ))

        # Simulate step-by-step
        for w in 1:num_windows
            curr_ovr, curr_val, curr_age, status = evolution_step(
                curr_ovr, curr_pot, curr_age, curr_val, w
            )
            
            push!(analysis_rows, (
                player_id = row.player_id, name = row.name, 
                window = w, age = curr_age, 
                ovr = round(curr_ovr, digits=1), value = round(curr_val, digits=2),
                status = status
            ))
        end
    end

    df_analysis = DataFrame(analysis_rows)
    CSV.write("data/processed/evolution_analysis.csv", df_analysis)
    println("Sequential analysis saved to data/processed/evolution_analysis.csv")
    return df_analysis
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

Creates a unified 'master_audit.csv' with separate columns for League Multiplier 
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
    CSV.write("data/processed/master_audit.csv", df_master)
    println("✅ Master audit saved with financial breakdown.")
    return df_master
end
