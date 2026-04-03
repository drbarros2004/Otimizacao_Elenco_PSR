using CSV, DataFrames, Dates, TOML

# Try to load optimization dependencies
const GUROBI_AVAILABLE = try
    using JuMP, Gurobi
    true
catch e
    false
end

# --- Import Dependencies (final explicit load order) ---
include("src/scenario_tree.jl")
include("src/player_dynamics.jl")
include("src/market_logic.jl")
include("src/data_loader.jl")
include("src/projections.jl")
include("src/model.jl")
include("src/config_parser.jl")
include("src/runner.jl")

# Check if the script is run directly from the terminal
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
