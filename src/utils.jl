using CSV, DataFrames, Dates, Random, Statistics

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
    get_free_agent_signing_multiplier(ovr::Int) -> Float64

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