using CSV, DataFrames, Dates, Random

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
    evolution_step(ovr::Float64, pot::Float64, age::Int, value::Float64, window_idx::Int)

Deterministic evolution logic where age directly influences the speed of growth and decay.
"""
function evolution_step(ovr::Float64, pot::Float64, age::Int, value::Float64, window_idx::Int)
    # 1. Age increments every 2 windows
    current_age = (window_idx > 0 && window_idx % 2 != 0) ? age + 1 : age
    
    delta = 0.0
    status = "Stable"

    if current_age < 30
        # --- GROWTH PHASE ---
        # Coefficient decreases linearly from ~0.15 at age 16 to ~0.02 at age 29
        # Formula: max(0.02, 0.25 - 0.008 * current_age)
        growth_coeff = max(0.02, 0.28 - (0.01 * current_age))
        
        # Growth is proportional to the potential gap
        gap = max(0.0, pot - ovr)
        delta = gap * growth_coeff
        status = "Growth"
    else
        # --- DECAY PHASE ---
        # Decay accelerates with age. 
        # A 30-year-old starts at -0.2 per window. 
        # Formula: -0.2 * (current_age - 29)^0.8 (non-linear acceleration)
        delta = -0.2 * (current_age - 29)
        status = "Decay"
    end
    
    # 2. Update OVR (Rounded to maintain Int stability in the map)
    new_ovr = clamp(ovr + delta, 40.0, 99.0)
    
    # 3. Update Market Value (Financial Dynamics)
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
    get_market_multiplier(...)
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