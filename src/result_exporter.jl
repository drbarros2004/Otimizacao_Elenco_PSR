"""
Result extraction and export utilities.
Transforms JuMP variables into tidy tables and persists CSV outputs.
"""

using CSV, DataFrames, JuMP

"""
    extract_deterministic_results(model, data; ...)

Builds deterministic result tables from solved model variables.
"""
function extract_deterministic_results(
    model::Model,
    data::ModelData;
    objective_value::Float64 = objective_value(model),
    solve_time::Float64 = 0.0,
    verbose::Bool = true
)
    J = data.players.player_id
    T = data.windows
    pos_groups = Dict(row.player_id => row.pos_group for row in eachrow(data.players))

    decision_rows = []
    for j in J, t in T
        x_val = round(Int, value(model[:x][j, t]))
        y_val = round(Int, value(model[:y][j, t]))
        b_val = round(Int, value(model[:buy][j, t]))
        s_val = round(Int, value(model[:sell][j, t]))

        if x_val == 1 || b_val == 1 || s_val == 1
            formation_scheme = get(data.formation_by_window, t, "default")
            push!(decision_rows, (
                player_id = j,
                window = t,
                formation_scheme = formation_scheme,
                in_squad = x_val,
                is_starter = y_val,
                starter_in_window = y_val,
                bought = b_val,
                sold = s_val,
                ovr = data.ovr_map[(j, t)],
                market_value = data.value_map[(j, t)],
                acquisition_cost = data.cost_map[(j, t)]
            ))
        end
    end

    df_decisions = DataFrame(decision_rows)

    budget_rows = [(
        window = t,
        cash_balance = value(model[:budget][t]),
        deficit = value(model[:budget_deficit][t])
    ) for t in T]
    df_budget = DataFrame(budget_rows)

    diagnostics_rows = []
    for t in T
        scheme_name = get(data.formation_by_window, t, "default")
        if !haskey(data.formation_catalog, scheme_name)
            error("Missing scheme '$scheme_name' for window $t in formation catalog.")
        end

        counts = data.formation_catalog[scheme_name]
        for (pos, required_count) in counts
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            starters_count = sum(round(Int, value(model[:y][j, t])) for j in players_in_pos)

            push!(diagnostics_rows, (
                window = t,
                formation_scheme = scheme_name,
                pos_group = pos,
                required_count = required_count,
                actual_starters = starters_count,
                slack_titular = 0.0
            ))
        end
    end

    df_formation_diagnostics = DataFrame(diagnostics_rows)

    verbose && println("✅ Solution extracted successfully!")

    return ModelResults(model, objective_value, solve_time, df_decisions, df_budget, df_formation_diagnostics)
end

"""
    tree_to_dataframe(data)

Flattens scenario tree metadata into a tidy DataFrame.
"""
function tree_to_dataframe(data::ModelDataStochastic)
    leaf_set = Set(data.leaf_nodes)
    rows = []

    for n in data.node_ids
        node = data.tree.nodes[n]
        parent = data.parent_by_node[n]

        push!(rows, (
            node_id = n,
            parent_id = isnothing(parent) ? missing : parent,
            stage = data.stage_by_node[n],
            branch_probability = node.branch_probability,
            cumulative_probability = data.probability_by_node[n],
            tactical_scheme = node.tactical_scheme,
            is_leaf = n in leaf_set,
            children_count = length(node.children_ids),
            path = join(string.(data.path_by_node[n]), "->")
        ))
    end

    return DataFrame(rows)
end

"""
    extract_stochastic_results(model, data; ...)

Builds stochastic node-indexed result tables from solved model variables.
"""
function extract_stochastic_results(
    model::Model,
    data::ModelDataStochastic;
    objective_value::Float64 = objective_value(model),
    solve_time::Float64 = 0.0,
    verbose::Bool = true
)
    J = Int.(data.players.player_id)
    N = data.node_ids
    pos_groups = Dict(Int(row.player_id) => String(row.pos_group) for row in eachrow(data.players))

    decision_rows = []
    for n in N
        parent = data.parent_by_node[n]
        parent_id = isnothing(parent) ? missing : parent
        stage_n = data.stage_by_node[n]
        prob_n = data.probability_by_node[n]
        node = data.tree.nodes[n]
        path_id = join(string.(data.path_by_node[n]), "->")

        for j in J
            x_val = round(Int, value(model[:x][j, n]))
            y_val = round(Int, value(model[:y][j, n]))
            b_val = round(Int, value(model[:buy][j, n]))
            s_val = round(Int, value(model[:sell][j, n]))

            if x_val == 1 || b_val == 1 || s_val == 1
                push!(decision_rows, (
                    node_id = n,
                    parent_id = parent_id,
                    stage = stage_n,
                    cumulative_probability = prob_n,
                    path_id = path_id,
                    tactical_scheme = node.tactical_scheme,
                    player_id = j,
                    in_squad = x_val,
                    is_starter = y_val,
                    starter_in_node = y_val,
                    bought = b_val,
                    sold = s_val,
                    ovr = data.ovr_node_map[(j, n)],
                    market_value = data.value_node_map[(j, n)],
                    acquisition_cost = data.cost_node_map[(j, n)],
                    wage = data.wage_node_map[(j, n)]
                ))
            end
        end
    end
    df_decisions = DataFrame(decision_rows)

    budget_rows = []
    for n in N
        parent = data.parent_by_node[n]
        parent_id = isnothing(parent) ? missing : parent

        push!(budget_rows, (
            node_id = n,
            parent_id = parent_id,
            stage = data.stage_by_node[n],
            cumulative_probability = data.probability_by_node[n],
            cash_balance = value(model[:budget][n]),
            deficit = value(model[:budget_deficit][n])
        ))
    end
    df_budget = DataFrame(budget_rows)

    diagnostics_rows = []
    for n in N
        stage_n = data.stage_by_node[n]
        node = data.tree.nodes[n]

        for (pos, required_count) in node.position_requirements
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            starters_count = sum(round(Int, value(model[:y][j, n])) for j in players_in_pos)

            push!(diagnostics_rows, (
                node_id = n,
                stage = stage_n,
                tactical_scheme = node.tactical_scheme,
                pos_group = pos,
                required_count = required_count,
                actual_starters = starters_count,
                slack_titular = 0.0
            ))
        end
    end
    df_formation_diagnostics = DataFrame(diagnostics_rows)
    df_tree = tree_to_dataframe(data)

    verbose && println("✅ Stochastic solution extracted successfully!")

    return StochasticModelResults(
        model,
        objective_value,
        solve_time,
        df_decisions,
        df_budget,
        df_tree,
        df_formation_diagnostics
    )
end

"""
    export_results(results, data, output_dir)

Exports deterministic optimization outputs.
"""
function export_results(results::ModelResults, data::ModelData, output_dir::String="output")
    mkpath(output_dir)

    player_meta = select(
        data.players,
        :player_id,
        :name,
        :pos_group,
        :club_name,
        :club_league_name
    )
    rename!(player_meta, :club_name => :origin_club, :club_league_name => :origin_league)

    df_output = leftjoin(results.squad_decisions, player_meta, on=:player_id)

    CSV.write("$output_dir/squad_decisions.csv", df_output)
    CSV.write("$output_dir/budget_evolution.csv", results.budget_evolution)
    CSV.write("$output_dir/formation_diagnostics.csv", results.formation_diagnostics)

    println("\n💾 Results exported to '$output_dir/':")
    println("   • squad_decisions.csv")
    println("   • budget_evolution.csv")
    println("   • formation_diagnostics.csv")

    return df_output
end

"""
    export_stochastic_results(results, data, output_dir)

Exports node-indexed stochastic optimization outputs.
"""
function export_stochastic_results(
    results::StochasticModelResults,
    data::ModelDataStochastic,
    output_dir::String="output"
)
    mkpath(output_dir)

    player_meta = select(
        data.players,
        :player_id,
        :name,
        :pos_group,
        :club_name,
        :club_league_name
    )
    rename!(player_meta, :club_name => :origin_club, :club_league_name => :origin_league)

    df_output = leftjoin(results.node_decisions, player_meta, on=:player_id)

    CSV.write("$output_dir/squad_decisions_nodes.csv", df_output)
    CSV.write("$output_dir/budget_evolution_nodes.csv", results.budget_evolution)
    CSV.write("$output_dir/tree_metadata.csv", results.tree_metadata)
    CSV.write("$output_dir/formation_diagnostics_nodes.csv", results.formation_diagnostics)

    println("\n💾 Stochastic results exported to '$output_dir/':")
    println("   • squad_decisions_nodes.csv")
    println("   • budget_evolution_nodes.csv")
    println("   • tree_metadata.csv")
    println("   • formation_diagnostics_nodes.csv")

    return df_output
end
