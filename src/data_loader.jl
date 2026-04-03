using CSV, DataFrames

# --- PATH CONFIGURATIONS (INGESTION SCOPE) ---
const RAW_BASE_PATH = "data/raw/player_stats.csv"
const RAW_CUSTOM_PATH = "data/raw/stats_brasileirao_with_correct_teams.csv"
const PROCESSED_OUTPUT_PATH = "data/processed/processed_player_data.csv"

# Column mapping from the SoFIFA scraper
const SCRAPER_COLUMNS = [
    :player_id, :name, :dob, :positions, :overall_rating,
    :potential, :value, :wage, :international_reputation,
    :club_name, :club_league_name
]

"""
Maps SoFIFA position strings to granular tactical position groups.
Returns the player's primary position group.
"""
function map_position_group(position_string::AbstractString)
    primary_pos = strip(split(String(position_string), ",")[1])

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
Loads raw player data from base and custom sources, merges them,
handles duplicates, and performs initial feature engineering (age calculation).
"""
function load_and_clean_data()
    println("Starting data ingestion using SoFIFA schema...")

    if !isfile(RAW_BASE_PATH) || !isfile(RAW_CUSTOM_PATH)
        error("Source files not found in data/raw/. Check your paths.")
    end

    df_base = CSV.read(RAW_BASE_PATH, DataFrame)
    df_custom = CSV.read(RAW_CUSTOM_PATH, DataFrame)

    # Remove Brazilian-team rows from base source to avoid unlicensed players.
    # Keep Brazilian teams only from the custom source.
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

    # The custom data doesn't include the league, so we hard-code it here.
    df_custom.club_league_name .= "Brazil Serie A"

    # Merge datasets via vertical concatenation.
    df_total = vcat(df_base, df_custom, cols=:union)

    # Remove duplicates by player_id.
    n_before = nrow(df_total)
    unique!(df_total, :player_id)
    println("✂️ Removed $(n_before - nrow(df_total)) duplicate records.")

    # Keep only relevant columns.
    df_cleaned = df_total[:, intersect(names(df_total), String.(SCRAPER_COLUMNS))]

    # Feature engineering.
    df_cleaned.age = calculate_age.(df_cleaned.dob)
    df_cleaned.pos_group = map_position_group.(df_cleaned.positions)

    # Null handling (imputation).
    df_cleaned.value = Float64.(coalesce.(df_cleaned.value, 0.0))
    df_cleaned.international_reputation = Float64.(coalesce.(df_cleaned.international_reputation, 1.0))
    df_cleaned.club_name = coalesce.(df_cleaned.club_name, "Unknown")

    mkpath(dirname(PROCESSED_OUTPUT_PATH))
    CSV.write(PROCESSED_OUTPUT_PATH, df_cleaned)

    println("Success! Total unique players: $(nrow(df_cleaned))")
    println("Generated 'age' column for future growth/decay projections.")

    return df_cleaned
end

"""
    filter_top_k_players_by_position(df::DataFrame, k::Int; protected_player_ids::Vector{Int}=Int[])

Keeps only the top-K players per position group based on overall quality proxies,
optionally preserving a set of protected players.
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
