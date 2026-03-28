"""
Result Analysis Script
======================

This script provides utilities to analyze optimization results and
extract meaningful insights about squad decisions.
"""

using CSV, DataFrames, Statistics

"""
    analyze_squad_decisions(decisions_path::String="output/squad_decisions.csv")

Analyzes squad decisions and prints a comprehensive report.

# What the outputs mean:
- **in_squad=1**: Player is in the roster (can be bench or starter)
- **is_starter=1**: Player is in the starting XI
- **bought=1**: Player was purchased in this window
- **sold=1**: Player was sold in this window
"""
function analyze_squad_decisions(decisions_path::String="output/squad_decisions.csv")
    if !isfile(decisions_path)
        error("Results file not found at $decisions_path. Run optimization first!")
    end

    df = CSV.read(decisions_path, DataFrame)

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
    export_starters_by_window(decisions_path::String, output_path::String="output/starters_summary.csv")

Creates a simplified CSV with just the starters for each window.
"""
function export_starters_by_window(decisions_path::String, output_path::String="output/starters_summary.csv")
    df = CSV.read(decisions_path, DataFrame)
    starters = filter(r -> r.is_starter == 1, df)

    # Sort for better readability
    sort!(starters, [:window, :pos_group, order(:ovr, rev=true)])

    CSV.write(output_path, starters)
    println("✅ Starters summary saved to $output_path")

    return starters
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
        # Main analysis
        analyze_squad_decisions()
        analyze_budget_evolution()

        # Export starters
        export_starters_by_window("output/squad_decisions.csv")

        println("\n" * "="^70)
        println("✅ ANALYSIS COMPLETE!")
        println("="^70)
        println("\nGenerated files:")
        println("   • output/squad_decisions.csv    - Full decision matrix")
        println("   • output/budget_evolution.csv   - Financial tracking")
        println("   • output/starters_summary.csv   - Starting XI per window")

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
