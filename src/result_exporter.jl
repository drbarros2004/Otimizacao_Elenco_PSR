"""
Result extraction and export utilities.
Transforms JuMP variables into tidy tables and persists CSV outputs.
"""

using CSV, DataFrames, JuMP

function _build_player_metadata(players::DataFrame, is_foreign_map::Dict{Int, Bool})
    meta_cols = Symbol[:player_id, :name, :pos_group, :club_name, :club_league_name]
    has_country = :country_name in Symbol.(names(players))

    if has_country
        push!(meta_cols, :country_name)
    end

    player_meta = select(players, meta_cols)
    rename!(player_meta, :club_name => :origin_club, :club_league_name => :origin_league)

    if has_country
        rename!(player_meta, :country_name => :nationality)
    else
        player_meta.nationality = fill(missing, nrow(player_meta))
    end

    player_meta.is_foreign = [get(is_foreign_map, Int(p_id), false) for p_id in player_meta.player_id]

    return player_meta
end

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
    J = Int.(data.players.player_id)
    T = data.windows
    pos_groups = Dict(Int(row.player_id) => String(row.pos_group) for row in eachrow(data.players))

    decision_rows = []
    for j in J, t in T
        x_val = round(Int, value(model[:x][j, t]))
        y_val = round(Int, value(model[:y][j, t]))
        b_val = round(Int, value(model[:buy][j, t]))
        s_val = round(Int, value(model[:sell][j, t]))

        if x_val == 1 || b_val == 1 || s_val == 1
            formation_scheme = get(data.formation_by_window, t, "default")
            ovr_now = data.ovr_map[(j, t)]
            ovr_prev = t > first(T) ? data.ovr_map[(j, t - 1)] : ovr_now
            push!(decision_rows, (
                player_id = j,
                window = t,
                formation_scheme = formation_scheme,
                in_squad = x_val,
                is_starter = y_val,
                starter_in_window = y_val,
                bought = b_val,
                sold = s_val,
                starter_allowed = 1,
                injured = 0,
                ovr = ovr_now,
                ovr_prev = ovr_prev,
                ovr_delta = ovr_now - ovr_prev,
                market_value = data.value_map[(j, t)],
                acquisition_cost = data.cost_map[(j, t)]
            ))
        end
    end

    df_decisions = DataFrame(decision_rows)

    budget_rows = []
    for t in T
        spent = sum(data.cost_map[(j, t)] * round(Int, value(model[:buy][j, t])) for j in J)
        sold = sum(data.value_map[(j, t)] * round(Int, value(model[:sell][j, t])) for j in J)
        buys = sum(round(Int, value(model[:buy][j, t])) for j in J)
        sells = sum(round(Int, value(model[:sell][j, t])) for j in J)

        push!(budget_rows, (
            window = t,
            cash_balance = millions_to_money(value(model[:budget][t])),
            deficit = millions_to_money(value(model[:budget_deficit][t])),
            transfer_spent = spent,
            transfer_sold = sold,
            buys_count = buys,
            sells_count = sells,
        ))
    end
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
                slack_titular = Float64(required_count - starters_count)
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
    node_ids = sort(collect(keys(data.tree.nodes)))
    leaf_set = Set(data.tree.leaf_nodes)
    rows = []

    for n in node_ids
        node = data.tree.nodes[n]
        parent = node.parent_id
        effective_scheme = get_effective_tactical_scheme(node)
        scenario_label = get(node.metadata, "manual_label", missing)

        push!(rows, (
            node_id = n,
            parent_id = isnothing(parent) ? missing : parent,
            stage = node.stage,
            branch_probability = node.branch_probability,
            cumulative_probability = node.cumulative_probability,
            tactical_scheme = effective_scheme,
            scenario_label = scenario_label,
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
    N = sort(collect(keys(data.tree.nodes)))
    pos_groups = Dict(Int(row.player_id) => String(row.pos_group) for row in eachrow(data.players))

    decision_rows = []
    for n in N
        node = data.tree.nodes[n]
        parent = node.parent_id
        parent_id = isnothing(parent) ? missing : parent
        stage_n = node.stage
        prob_n = node.cumulative_probability
        effective_scheme = get_effective_tactical_scheme(node)
        path_id = join(string.(data.path_by_node[n]), "->")

        for j in J
            x_val = round(Int, value(model[:x][j, n]))
            y_val = round(Int, value(model[:y][j, n]))
            b_val = round(Int, value(model[:buy][j, n]))
            s_val = round(Int, value(model[:sell][j, n]))

            if x_val == 1 || b_val == 1 || s_val == 1
                ovr_now = data.ovr_node_map[(j, n)]
                ovr_prev = if isnothing(parent)
                    ovr_now
                else
                    data.ovr_node_map[(j, parent::Int)]
                end

                parent_buy = if isnothing(parent)
                    0
                else
                    round(Int, value(model[:buy][j, parent::Int]))
                end
                parent_sell = if isnothing(parent)
                    0
                else
                    round(Int, value(model[:sell][j, parent::Int]))
                end

                starter_allowed = data.starter_allowed_map[(j, n)]
                push!(decision_rows, (
                    node_id = n,
                    parent_id = parent_id,
                    stage = stage_n,
                    cumulative_probability = prob_n,
                    path_id = path_id,
                    tactical_scheme = effective_scheme,
                    player_id = j,
                    in_squad = x_val,
                    is_starter = y_val,
                    starter_in_node = y_val,
                    bought = b_val,
                    sold = s_val,
                    bought_in_parent = parent_buy,
                    sold_in_parent = parent_sell,
                    bought_from_root = (!isnothing(parent) && parent::Int == data.tree.root_id && parent_buy == 1) ? 1 : 0,
                    starter_allowed = starter_allowed,
                    injured = starter_allowed == 1 ? 0 : 1,
                    ovr = ovr_now,
                    ovr_prev = ovr_prev,
                    ovr_delta = ovr_now - ovr_prev,
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
        node = data.tree.nodes[n]
        parent = node.parent_id
        parent_id = isnothing(parent) ? missing : parent
        spent = sum(data.cost_node_map[(j, n)] * round(Int, value(model[:buy][j, n])) for j in J)
        sold = sum(data.value_node_map[(j, n)] * round(Int, value(model[:sell][j, n])) for j in J)
        buys = sum(round(Int, value(model[:buy][j, n])) for j in J)
        sells = sum(round(Int, value(model[:sell][j, n])) for j in J)
        foreign_squad_count = sum(
            round(Int, value(model[:x][j, n])) * (get(data.is_foreign_map, j, false) ? 1 : 0)
            for j in J
        )
        foreign_starter_count = sum(
            round(Int, value(model[:y][j, n])) * (get(data.is_foreign_map, j, false) ? 1 : 0)
            for j in J
        )

        push!(budget_rows, (
            node_id = n,
            parent_id = parent_id,
            stage = node.stage,
            cumulative_probability = node.cumulative_probability,
            cash_balance = millions_to_money(value(model[:budget][n])),
            deficit = millions_to_money(value(model[:budget_deficit][n])),
            transfer_spent = spent,
            transfer_sold = sold,
            buys_count = buys,
            sells_count = sells,
            foreign_squad_count = foreign_squad_count,
            foreign_starter_count = foreign_starter_count,
            foreign_excess = value(model[:excess_foreigners][n]),
        ))
    end
    df_budget = DataFrame(budget_rows)

    diagnostics_rows = []
    for n in N
        node = data.tree.nodes[n]
        stage_n = node.stage
        effective_scheme = get_effective_tactical_scheme(node)

        for (pos, required_count) in node.position_requirements
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            starters_count = sum(round(Int, value(model[:y][j, n])) for j in players_in_pos)

            push!(diagnostics_rows, (
                node_id = n,
                stage = stage_n,
                tactical_scheme = effective_scheme,
                pos_group = pos,
                required_count = required_count,
                actual_starters = starters_count,
                slack_titular = Float64(required_count - starters_count)
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

    player_meta = _build_player_metadata(data.players, data.is_foreign_map)

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

    player_meta = _build_player_metadata(data.players, data.is_foreign_map)

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
