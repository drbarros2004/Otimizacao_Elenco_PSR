using DataFrames, Statistics

"""
Returns a market multiplier based on origin league reputation and international reputation.
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

    valid_mask = (coalesce.(df[!, wage_col], 0.0) .> 0) .& (coalesce.(df[!, ovr_col], 0) .>= 70)
    valid_players = df[valid_mask, :]

    sort!(valid_players, ovr_col, rev=true)

    sample_size = min(2000, nrow(valid_players))
    top_players = first(valid_players, sample_size)

    println("   Using $(sample_size) top players for wage inference")

    wage_by_ovr = Dict{Int, Float64}()

    for ovr in 70:99
        ovr_players = top_players[top_players[!, ovr_col] .== ovr, :]
        if nrow(ovr_players) > 0
            wage_by_ovr[ovr] = mean(skipmissing(Float64.(ovr_players[!, wage_col])))
        end
    end

    filled_wage_by_ovr = Dict{Int, Float64}()

    for ovr in 70:99
        if haskey(wage_by_ovr, ovr)
            filled_wage_by_ovr[ovr] = wage_by_ovr[ovr]
        else
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

            if !isnothing(lower_ovr) && !isnothing(upper_ovr)
                ratio = (ovr - lower_ovr) / (upper_ovr - lower_ovr)
                filled_wage_by_ovr[ovr] = wage_by_ovr[lower_ovr] + ratio * (wage_by_ovr[upper_ovr] - wage_by_ovr[lower_ovr])
            elseif !isnothing(lower_ovr)
                filled_wage_by_ovr[ovr] = wage_by_ovr[lower_ovr] * (1.05 ^ (ovr - lower_ovr))
            elseif !isnothing(upper_ovr)
                filled_wage_by_ovr[ovr] = wage_by_ovr[upper_ovr] * (0.95 ^ (upper_ovr - ovr))
            else
                filled_wage_by_ovr[ovr] = Float64(ovr * 2000)
            end
        end
    end

    for ovr in 40:69
        filled_wage_by_ovr[ovr] = Float64(ovr * 1000)
    end

    println("   ✅ Wage inference complete for OVR 40-99")
    println("   Sample wages: OVR 70 = €$(round(Int, filled_wage_by_ovr[70])), OVR 80 = €$(round(Int, filled_wage_by_ovr[80])), OVR 90 = €$(round(Int, filled_wage_by_ovr[90]))")

    return filled_wage_by_ovr
end

"""
Returns the signing bonus multiplier for free agents based on their OVR.
Higher quality free agents demand significantly higher signing bonuses.
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
        return 2.0
    end
end
