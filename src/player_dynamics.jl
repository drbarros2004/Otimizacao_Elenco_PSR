using Dates

"""
Calculates the player's age based on the Date of Birth (dob).
If the value is missing or invalid, returns a default age of 25.
"""
function calculate_age(dob_value)
    if ismissing(dob_value)
        return 25
    end

    try
        birth_date = if dob_value isa Date
            dob_value
        else
            Date(strip(string(dob_value)))
        end

        today = Date(2026, 03, 26)

        return year(today) - year(birth_date) - (monthday(today) < monthday(birth_date) ? 1 : 0)
    catch e
        @warn "Failed to calculate age for value: $dob_value. Error: $e"
        return 25
    end
end

"""
Deterministic evolution logic where age directly influences the speed of growth and decay.
"""
function evolution_step(ovr::Float64, pot::Float64, age::Int, value::Float64, window_idx::Int)
    current_age = (window_idx > 0 && window_idx % 2 != 0) ? age + 1 : age

    growth_intercept = 0.34
    growth_age_slope = 0.0095
    min_growth_coeff = 0.065
    youth_bonus_age_limit = 29
    youth_bonus_multiplier = 1.65

    delta = 0.0
    status = "Stable"

    if current_age < 30
        growth_coeff = max(min_growth_coeff, growth_intercept - (growth_age_slope * current_age))
        youth_bonus = current_age <= youth_bonus_age_limit ? youth_bonus_multiplier : 1.0

        gap = max(0.0, pot - ovr)
        delta = gap * growth_coeff * youth_bonus
        status = "Growth"
    else
        delta = -0.2 * (current_age - 29)
        status = "Decay"
    end

    new_ovr = clamp(ovr + delta, 40.0, 99.0)

    age_value_factor = clamp(1.4 - (0.015 * current_age), 0.7, 1.2)
    new_value = value * (1 + (delta / 100.0)) * age_value_factor
    new_value = max(new_value, 0.5)

    return round(new_ovr), new_value, current_age, status
end
