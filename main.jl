using CSV, DataFrames, Dates, TOML

# --- 1. Import Dependencies ---
include("src/data_loader.jl")

# --- 2. Configuration ---
# You can change these values here without touching the internal logic
NUM_WINDOWS = 6  # 3 years of simulation (2 windows per year)

function run_pipeline()
    println("🚀 Starting PSR Squad Optimization Pipeline...")
    println("="^50)

    # STEP 1: Data Ingestion & Cleaning
    # Loads from raw CSVs, merges, calculates age and saves processed data
    df = load_and_clean_data()

    # STEP 2: Projections (Deterministic Logic)
    # Generates OVR and Selling Value maps
    ovr_map, value_map = generate_projections(df, NUM_WINDOWS)

    # STEP 3: Market Reputation (Acquisition Costs)
    # Applies TOML weights to generate the Buying Cost map
    cost_map = generate_cost_map(df, value_map, NUM_WINDOWS)

    # STEP 4: Unified Analytical Export
    export_analysis(df, ovr_map, value_map, cost_map, NUM_WINDOWS)

    println("="^50)
    println("✅ Pipeline finished! Master report saved in 'data/processed/master_audit.csv'.")

    return df, ovr_map, value_map, cost_map
end

# Check if the script is run directly from the terminal
if abspath(PROGRAM_FILE) == @__FILE__
    run_pipeline()
end