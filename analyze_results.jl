"""
Result Analysis Script
======================

This script provides utilities to analyze optimization results and
extract meaningful insights about squad decisions.
"""

using CSV, DataFrames, Statistics

const DET_DECISIONS_PATH = "output/squad_decisions.csv"
const DET_BUDGET_PATH = "output/budget_evolution.csv"
const STOCH_DECISIONS_PATH = "output/squad_decisions_nodes.csv"
const STOCH_BUDGET_PATH = "output/budget_evolution_nodes.csv"

_files_exist(paths::Vector{String}) = all(isfile, paths)

function _latest_mtime(paths::Vector{String}) # isso é a melhor forma de fazer isso?
    return maximum((isfile(p) ? Float64(stat(p).mtime) : 0.0) for p in paths)
end

function _parse_analysis_mode(args::Vector{String})
    if "--deterministic" in args
        return :deterministic
    elseif "--stochastic" in args
        return :stochastic
    else
        return :auto
    end
end

function _infer_analysis_mode(preferred::Symbol=:auto)
    det_paths = [DET_DECISIONS_PATH, DET_BUDGET_PATH]
    stoch_paths = [STOCH_DECISIONS_PATH, STOCH_BUDGET_PATH]

    det_ready = _files_exist(det_paths)
    stoch_ready = _files_exist(stoch_paths)

    if preferred == :deterministic
        det_ready || error("Deterministic files are missing. Run deterministic optimization first.")
        return :deterministic
    elseif preferred == :stochastic
        stoch_ready || error("Stochastic files are missing. Run stochastic optimization first.")
        return :stochastic
    end

    if det_ready && stoch_ready
        # Auto mode: choose the most recent result set to avoid stale-file confusion.
        return _latest_mtime(det_paths) >= _latest_mtime(stoch_paths) ? :deterministic : :stochastic
    elseif det_ready
        return :deterministic
    elseif stoch_ready
        return :stochastic
    else
        error("No result files found. Run optimization first.")
    end
end

"""
    analyze_squad_decisions(
        decisions_path::String="output/squad_decisions.csv",
        formation_diagnostics_path::String="output/formation_diagnostics.csv"
    )

Analyzes squad decisions and prints a comprehensive report.

# What the outputs mean:
- **in_squad=1**: Player is in the roster (can be bench or starter)
- **is_starter=1**: Player is in the starting XI
- **bought=1**: Player was purchased in this window
- **sold=1**: Player was sold in this window
"""
function analyze_squad_decisions(
    decisions_path::String="output/squad_decisions.csv",
    formation_diagnostics_path::String="output/formation_diagnostics.csv"
)
    if !isfile(decisions_path)
        error("Results file not found at $decisions_path. Run optimization first!")
    end

    df = CSV.read(decisions_path, DataFrame)
    df_formation = isfile(formation_diagnostics_path) ? CSV.read(formation_diagnostics_path, DataFrame) : nothing

    println("\n" * "="^70)
    println("SQUAD OPTIMIZATION RESULTS ANALYSIS")
    println("="^70)

    # Summary statistics
    total_windows = maximum(df.window)
    total_players = length(unique(df.player_id))

    println("\n📊 OVERVIEW")
    println("   • Total windows analyzed: $(total_windows + 1)")
    println("   • Total unique players involved: $total_players")

    # Analyze each window
    for window in 0:total_windows
        df_window = filter(r -> r.window == window, df)

        println("\n" * "─"^70)
        println("📅 WINDOW $window")
        println("─"^70)

        if :formation_scheme in names(df_window) && nrow(df_window) > 0
            active_scheme = first(df_window.formation_scheme)
            println("🧠 Active tactical scheme: $active_scheme")
        end

        # Squad composition
        squad = filter(r -> r.in_squad == 1, df_window)
        starters = filter(r -> r.is_starter == 1, df_window)

        println("\n🏟️  SQUAD COMPOSITION ($(nrow(squad)) players)")

        # Position breakdown
        for pos in sort(unique(skipmissing(df.pos_group)))
            squad_pos = filter(r -> r.pos_group == pos, squad)
            starters_pos = filter(r -> r.pos_group == pos, starters)
            println("   $pos: $(nrow(squad_pos)) total ($(nrow(starters_pos)) starters)")
        end

        # Starting XI
        println("\n⭐ STARTING XI ($(nrow(starters)) players):")
        starters_sorted = sort(starters, [:pos_group, order(:ovr, rev=true)])
        for row in eachrow(starters_sorted)
            println("   $(row.pos_group) | $(rpad(row.name, 30)) | OVR: $(row.ovr)")
        end

        # Average OVR
        avg_ovr_starters = mean(starters.ovr)
        avg_ovr_squad = mean(squad.ovr)
        println("\n📈 QUALITY METRICS:")
        println("   • Average OVR (Starters): $(round(avg_ovr_starters, digits=2))")
        println("   • Average OVR (Full Squad): $(round(avg_ovr_squad, digits=2))")

        if !isnothing(df_formation)
            df_form_window = filter(r -> r.window == window, df_formation)
            if nrow(df_form_window) > 0
                println("\n🧩 TACTICAL CONSTRAINTS:")
                for row in eachrow(sort(df_form_window, :pos_group))
                    println("   • $(row.pos_group): $(row.actual_starters) starters | required $(row.required_count) | slack $(round(row.slack_titular, digits=3))")
                end
            end
        end

        # Transactions
        if window > 0
            bought = filter(r -> r.bought == 1, df_window)
            sold = filter(r -> r.sold == 1, df_window)

            if nrow(bought) > 0 || nrow(sold) > 0
                println("\n💰 TRANSACTIONS:")

                if nrow(bought) > 0
                    println("\n   🛒 PURCHASES ($(nrow(bought))):")
                    for row in eachrow(bought)
                        cost_m = round(row.acquisition_cost / 1e6, digits=1)
                        println("      • $(rpad(row.name, 30)) | $(row.pos_group) | OVR: $(row.ovr) | €$(cost_m)M")
                    end
                    total_spent = sum(bought.acquisition_cost)
                    println("      TOTAL SPENT: €$(round(total_spent/1e6, digits=1))M")
                end

                if nrow(sold) > 0
                    println("\n   💸 SALES ($(nrow(sold))):")
                    for row in eachrow(sold)
                        value_m = round(row.market_value / 1e6, digits=1)
                        println("      • $(rpad(row.name, 30)) | $(row.pos_group) | OVR: $(row.ovr) | €$(value_m)M")
                    end
                    total_revenue = sum(sold.market_value)
                    println("      TOTAL REVENUE: €$(round(total_revenue/1e6, digits=1))M")
                end
            else
                println("\n💰 TRANSACTIONS: None")
            end
        end
    end

    # Overall transaction summary
    println("\n" * "="^70)
    println("📋 OVERALL TRANSACTION SUMMARY")
    println("="^70)

    total_bought = filter(r -> r.bought == 1, df)
    total_sold = filter(r -> r.sold == 1, df)

    println("\n   Total players bought: $(nrow(total_bought))")
    println("   Total players sold: $(nrow(total_sold))")

    if nrow(total_bought) > 0
        println("   Total spent: €$(round(sum(total_bought.acquisition_cost)/1e6, digits=1))M")
    end
    if nrow(total_sold) > 0
        println("   Total revenue: €$(round(sum(total_sold.market_value)/1e6, digits=1))M")
    end

    return df
end

"""
    analyze_budget_evolution(budget_path::String="output/budget_evolution.csv")

Analyzes budget evolution over time.
"""
function analyze_budget_evolution(budget_path::String="output/budget_evolution.csv")
    if !isfile(budget_path)
        error("Budget file not found at $budget_path. Run optimization first!")
    end

    df = CSV.read(budget_path, DataFrame)

    println("\n" * "="^70)
    println("BUDGET EVOLUTION ANALYSIS")
    println("="^70)

    for row in eachrow(df)
        balance_m = round(row.cash_balance / 1e6, digits=1)
        deficit_m = round(row.deficit / 1e6, digits=1)

        status = deficit_m > 0 ? "⚠️  DEFICIT" : "✅ OK"
        println("\n   Window $(row.window): €$(balance_m)M | Deficit: €$(deficit_m)M | $status")
    end

    # Overall financial health
    final_balance = df[end, :cash_balance]
    total_deficit = sum(df.deficit)

    println("\n" * "─"^70)
    println("   Final Cash Balance: €$(round(final_balance/1e6, digits=1))M")
    println("   Total Budget Violations: €$(round(total_deficit/1e6, digits=1))M")

    if total_deficit > 0
        println("\n   ⚠️  WARNING: Budget constraints were violated!")
    else
        println("\n   ✅ All budget constraints satisfied!")
    end

    return df
end

"""
    compare_windows(decisions_path::String, window1::Int, window2::Int)

Compares squad composition between two windows to see what changed.
"""
function compare_windows(decisions_path::String, window1::Int, window2::Int)
    df = CSV.read(decisions_path, DataFrame)

    df_w1 = filter(r -> r.window == window1 && r.in_squad == 1, df)
    df_w2 = filter(r -> r.window == window2 && r.in_squad == 1, df)

    players_w1 = Set(df_w1.player_id)
    players_w2 = Set(df_w2.player_id)

    departed = setdiff(players_w1, players_w2)
    arrived = setdiff(players_w2, players_w1)
    stayed = intersect(players_w1, players_w2)

    println("\n" * "="^70)
    println("SQUAD CHANGES: Window $window1 → Window $window2")
    println("="^70)

    println("\n   Players who stayed: $(length(stayed))")
    println("   Players who left: $(length(departed))")
    println("   New arrivals: $(length(arrived))")

    if length(departed) > 0
        println("\n   ❌ DEPARTED:")
        departed_df = filter(r -> r.player_id in departed && r.window == window1, df)
        for row in eachrow(departed_df)
            println("      • $(row.name) ($(row.pos_group), OVR: $(row.ovr))")
        end
    end

    if length(arrived) > 0
        println("\n   ✅ ARRIVED:")
        arrived_df = filter(r -> r.player_id in arrived && r.window == window2, df)
        for row in eachrow(arrived_df)
            println("      • $(row.name) ($(row.pos_group), OVR: $(row.ovr))")
        end
    end
end

"""
    analyze_stochastic_decisions(
        decisions_path::String="output/squad_decisions_nodes.csv",
        tree_path::String="output/tree_metadata.csv"
    )

Analyzes node-indexed stochastic decisions.
"""
function analyze_stochastic_decisions(
    decisions_path::String="output/squad_decisions_nodes.csv",
    tree_path::String="output/tree_metadata.csv"
)
    if !isfile(decisions_path)
        error("Stochastic decisions file not found at $decisions_path. Run stochastic optimization first!")
    end

    df = CSV.read(decisions_path, DataFrame)
    df_tree = isfile(tree_path) ? CSV.read(tree_path, DataFrame) : nothing

    println("\n" * "="^70)
    println("STOCHASTIC NODE-INDEXED RESULTS ANALYSIS")
    println("="^70)

    total_nodes = length(unique(df.node_id))
    total_stages = maximum(df.stage)
    total_players = length(unique(df.player_id))

    println("\n📊 OVERVIEW")
    println("   • Total nodes analyzed: $total_nodes")
    println("   • Total stages analyzed: $(total_stages + 1)")
    println("   • Total unique players involved: $total_players")

    root_candidates = filter(r -> r.parent_id === missing, eachrow(df))
    if !isempty(root_candidates)
        root_node = first(root_candidates).node_id
        root_df = filter(r -> r.node_id == root_node && r.in_squad == 1, df)
        println("\n🎯 HERE-AND-NOW (ROOT NODE = $root_node)")
        println("   • Players in squad: $(nrow(root_df))")
        println("   • Average OVR: $(round(mean(root_df.ovr), digits=2))")
    end

    if !isnothing(df_tree)
        leaf_mask = map(x -> Bool(x), coalesce.(df_tree.is_leaf, false))
        leaf_nodes = df_tree[leaf_mask, :]
        if nrow(leaf_nodes) > 0
            prob_sum = sum(Float64.(leaf_nodes.cumulative_probability))
            println("\n🌳 TREE SANITY")
            println("   • Leaf nodes: $(nrow(leaf_nodes))")
            println("   • Sum of leaf probabilities: $(round(prob_sum, digits=6))")
        end
    end

    println("\n📌 CONTINGENCY SNAPSHOT")
    sold_df = filter(r -> r.sold == 1, df)
    bought_df = filter(r -> r.bought == 1, df)
    println("   • Total sell actions across nodes: $(nrow(sold_df))")
    println("   • Total buy actions across nodes: $(nrow(bought_df))")

    return df
end

"""
    analyze_stochastic_budget(budget_path::String="output/budget_evolution_nodes.csv")

Analyzes node-indexed budget evolution and stage-wise expected cash.
"""
function analyze_stochastic_budget(budget_path::String="output/budget_evolution_nodes.csv")
    if !isfile(budget_path)
        error("Stochastic budget file not found at $budget_path. Run stochastic optimization first!")
    end

    df = CSV.read(budget_path, DataFrame)

    println("\n" * "="^70)
    println("STOCHASTIC BUDGET ANALYSIS")
    println("="^70)

    stages = sort(unique(df.stage))
    for stage in stages
        df_stage = filter(r -> r.stage == stage, df)
        expected_cash = sum(df_stage.cash_balance .* df_stage.cumulative_probability)
        worst_cash = minimum(df_stage.cash_balance)
        best_cash = maximum(df_stage.cash_balance)

        println("\n   Stage $stage")
        println("   • Expected cash: €$(round(expected_cash/1e6, digits=2))M")
        println("   • Worst-case cash: €$(round(worst_cash/1e6, digits=2))M")
        println("   • Best-case cash: €$(round(best_cash/1e6, digits=2))M")
    end

    return df
end

# =============================================================================
# QUICK START
# =============================================================================

"""
Run this after your optimization completes to see all results!
"""
function quick_analysis()
    println("\n🔍 RUNNING COMPREHENSIVE ANALYSIS...")

    try
        mode = _infer_analysis_mode(_parse_analysis_mode(ARGS))

        if mode == :stochastic
            println("\nMode selected: stochastic")
            analyze_stochastic_decisions()
            analyze_stochastic_budget()
        else
            println("\nMode selected: deterministic")
            analyze_squad_decisions()
            analyze_budget_evolution()
        end

        println("\n" * "="^70)
        println("✅ ANALYSIS COMPLETE!")
        println("="^70)
        println("\nGenerated files:")
        println("   • output/squad_decisions.csv    - Full decision matrix")
        println("   • output/budget_evolution.csv   - Financial tracking")
        println("   • output/formation_diagnostics.csv - Tactical diagnostics")
        println("   • output/squad_decisions_nodes.csv - Node-indexed decision matrix")
        println("   • output/budget_evolution_nodes.csv - Node-indexed financial tracking")
        println("   • output/tree_metadata.csv - Scenario tree structure")
        println("   • output/formation_diagnostics_nodes.csv - Node-indexed tactical diagnostics")

    catch e
        println("\n❌ Error during analysis: $e")
        println("\nMake sure you've run the optimization first:")
        println("   julia main.jl")
    end
end

# Auto-run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    quick_analysis()
end
