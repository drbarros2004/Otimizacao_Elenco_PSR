"""
Stochastic Node-Based Squad Optimization Model (Pantuso-style).
"""

function build_stochastic_squad_optimization_model(
    data::ModelDataStochastic,
    params::ModelParameters;
    verbose::Bool=true
)
    verbose && println("\n Building Stochastic Node-Based Squad Optimization Model...")

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", 600)
    set_optimizer_attribute(model, "MIPGap", 0.01)
    !verbose && set_optimizer_attribute(model, "OutputFlag", 0)

    # Extract data
    players = data.players
    J = Int.(players.player_id)
    N = data.node_ids
    root_id = data.root_id
    pos_groups = Dict(Int(row.player_id) => String(row.pos_group) for row in eachrow(players))

    formation_positions = String[]
    for limits in values(data.formation_catalog)
        append!(formation_positions, collect(keys(limits)))
    end
    formation_positions = unique(sort(formation_positions))

    if isempty(formation_positions)
        error("No tactical positions found in stochastic formation catalog.")
    end

    # Chemistry pairs are kept based on initial squad for tractability.
    pairs = generate_player_pairs(data.initial_squad)
    verbose && println("  ├─ Chemistry pairs: $(length(pairs))")

    # ==== DECISION VARIABLES ====
    verbose && println("  ├─ Creating node-indexed decision variables...")

    @variable(model, x[j in J, n in N], Bin)
    @variable(model, y[j in J, n in N], Bin)
    @variable(model, buy[j in J, n in N], Bin)
    @variable(model, sell[j in J, n in N], Bin)
    @variable(model, budget[n in N] >= 0)
    @variable(model, budget_deficit[n in N] >= 0)

    # Chemistry variables
    @variable(model, ParElenco[(i,j) in pairs, n in N], Bin)
    @variable(model, ParTitular[(i,j) in pairs, n in N], Bin)
    @variable(model, 0 <= Quimica[(i,j) in pairs, n in N] <= S_MAX)

    # Soft constraint slacks
    @variable(model, slack_posicao[p in formation_positions, n in N] >= 0)
    @variable(model, slack_salario[n in N] >= 0)

    # ==== INITIAL CONDITIONS (ROOT NODE) ====
    verbose && println("  ├─ Root-node initial conditions...")

    @constraint(model, budget[root_id] == params.initial_budget)

    for j in J
        initial_presence = j in data.initial_squad ? 1 : 0
        @constraint(model, x[j, root_id] == initial_presence)
        # Keep deterministic timing parity: no transaction in initial state.
        @constraint(model, buy[j, root_id] == 0)
        @constraint(model, sell[j, root_id] == 0)
    end

    for (i, j) in pairs
        if i in data.initial_squad && j in data.initial_squad
            @constraint(model, Quimica[(i,j), root_id] == S_INICIAL)
        else
            @constraint(model, Quimica[(i,j), root_id] == 0)
        end
    end

    # ==== SQUAD FLOW (PARENT -> CHILD) ====
    verbose && println("  ├─ Squad flow constraints (parent-child timing)...")

    for n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end

        p = parent::Int
        for j in J
            @constraint(model, x[j, n] == x[j, p] + buy[j, p] - sell[j, p])
        end
    end

    # ==== BUDGET DYNAMICS (PARENT -> CHILD) ====
    verbose && println("  ├─ Budget constraints by node...")

    for n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int

        stage_n = data.stage_by_node[n]
        revenue = (stage_n % 2 == 0) ? params.seasonal_revenue : 0.0

        signing_costs = AffExpr(0.0)
        for j in J
            cost = data.cost_node_map[(j, n)]
            wage = data.wage_node_map[(j, n)]
            ovr = data.ovr_node_map[(j, n)]
            is_free_agent = (cost < 100_000)

            signing_cost = if is_free_agent
                multiplier = get_free_agent_signing_multiplier(ovr)
                wage * 52 * params.signing_bonus_rate * multiplier
            else
                wage * 52 * params.signing_bonus_rate
            end

            add_to_expression!(signing_costs, signing_cost * buy[j, p])
        end

        @constraint(model,
            budget[n] == budget[p]
                - sum((1 + params.transaction_cost_buy) * data.cost_node_map[(j,n)] * buy[j,p] for j in J)
                - signing_costs
                + sum((1 - params.transaction_cost_sell) * data.value_node_map[(j,n)] * sell[j,p] for j in J)
                + revenue
                + budget_deficit[n]
        )
    end

    # ==== SQUAD SIZE ====
    verbose && println("  ├─ Squad size constraints by node...")

    for n in N
        @constraint(model, sum(x[j,n] for j in J) <= params.max_squad_size)
        @constraint(model, sum(x[j,n] for j in J) >= params.min_squad_size)
    end

    # ==== SALARY CAP (SOFT) ====
    verbose && println("  ├─ Salary cap constraints (soft, node-indexed)...")

    initial_payroll = sum(data.wage_node_map[(j, root_id)] for j in data.initial_squad)
    salary_cap_per_node = initial_payroll * params.salary_cap_multiplier_initial * params.salary_cap_window_factor

    verbose && println("  │  Initial payroll baseline: €$(round(initial_payroll / 1e6, digits=2))M")
    verbose && println("  │  Salary cap per node: €$(round(salary_cap_per_node / 1e6, digits=2))M")

    for n in N
        @constraint(model,
            sum(data.wage_node_map[(j,n)] * x[j,n] for j in J) <= salary_cap_per_node + slack_salario[n]
        )
    end

    # ==== FORMATION (NODE-DEPENDENT N_{r,n}) ====
    verbose && println("  ├─ Tactical formation constraints by node (N_{r,n})...")

    for n in N
        @constraint(model, sum(y[j,n] for j in J) == 11)

        for pos in formation_positions
            required_count = get(data.position_requirements_map, (pos, n), 0)
            players_in_pos = [j for j in J if pos_groups[j] == pos]

            @constraint(model,
                sum(y[j,n] for j in players_in_pos) == required_count
            )
        end
    end

    # ==== STARTER ELIGIBILITY + INJURY SANDBOX ====
    verbose && println("  ├─ Starter and sell availability constraints...")

    for j in J, n in N
        starter_allowed = data.starter_allowed_map[(j, n)]
        sell_allowed = data.sell_allowed_map[(j, n)]

        @constraint(model, y[j,n] <= x[j,n] * starter_allowed)
        @constraint(model, sell[j,n] <= sell_allowed)
    end

    # ==== TRANSACTION EXCLUSIVITY ====
    verbose && println("  ├─ Transaction exclusivity by node...")

    for j in J, n in N
        @constraint(model, buy[j,n] + sell[j,n] <= 1)
    end

    # ==== CHEMISTRY LINEARIZATION ====
    verbose && println("  ├─ Chemistry linearization by node...")

    for (i, j) in pairs, n in N
        @constraint(model, ParElenco[(i,j), n] <= x[i, n])
        @constraint(model, ParElenco[(i,j), n] <= x[j, n])
        @constraint(model, ParElenco[(i,j), n] >= x[i,n] + x[j,n] - 1)

        @constraint(model, ParTitular[(i,j), n] <= y[i, n])
        @constraint(model, ParTitular[(i,j), n] <= y[j, n])
        @constraint(model, ParTitular[(i,j), n] >= y[i,n] + y[j,n] - 1)
    end

    # ==== CHEMISTRY DYNAMICS (PARENT -> CHILD) ====
    verbose && println("  ├─ Chemistry parent-child dynamics...")

    for (i, j) in pairs, n in N
        if n == root_id
            continue
        end

        parent = data.parent_by_node[n]
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int
        M = S_MAX

        @constraint(model,
            Quimica[(i,j), n] <= Quimica[(i,j), p] + BONUS_INCREMENTO + M * (1 - ParElenco[(i,j), n])
        )
        @constraint(model,
            Quimica[(i,j), n] >= Quimica[(i,j), p] + BONUS_INCREMENTO - M * (1 - ParElenco[(i,j), n])
        )

        @constraint(model,
            Quimica[(i,j), n] <= params.decay_quimica * Quimica[(i,j), p] + M * ParElenco[(i,j), n]
        )
        @constraint(model,
            Quimica[(i,j), n] >= params.decay_quimica * Quimica[(i,j), p] - M * ParElenco[(i,j), n]
        )
    end

    # ==== EXPECTED-VALUE OBJECTIVE (NODE-INDEXED) ====
    verbose && println("  ├─ Building expected-value objective...")

    obj_terms = AffExpr(0.0)

    non_root_nodes = [n for n in N if n != root_id]

    for n in non_root_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        for j in J
            score_mercado = (
                params.weight_quality * data.ovr_node_map[(j,n)] +
                params.weight_potential * data.growth_potential_node_map[(j,n)]
            )
            add_to_expression!(obj_terms, prob_n * params.peso_asset * score_mercado * x[j,n])

            score_tatico = data.ovr_node_map[(j,n)] * 1.0
            add_to_expression!(obj_terms, prob_n * params.peso_performance * score_tatico * y[j,n])

            add_to_expression!(obj_terms, -prob_n * params.friction_penalty * buy[j,n])
        end

        add_to_expression!(obj_terms, -prob_n * params.salary_cap_penalty * slack_salario[n])
        add_to_expression!(obj_terms, -prob_n * P_CAIXA * budget_deficit[n])
    end

    for n in non_root_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        chemistry_multiplier = get(data.chemistry_multiplier_map, n, 1.0)
        for (i, j) in pairs
            add_to_expression!(obj_terms, prob_n * params.bonus_entrosamento * chemistry_multiplier * Quimica[(i,j), n])
        end
    end

    # Terminal value is only accounted for leaf nodes, weighted by cumulative probability.
    for n in data.leaf_nodes
        prob_n = get(data.probability_by_node, n, 0.0)
        if prob_n <= 0.0
            continue
        end

        valor_elenco_final = sum(data.value_node_map[(j, n)] * x[j, n] for j in J)
        add_to_expression!(obj_terms, prob_n * 0.001 * valor_elenco_final)
        add_to_expression!(obj_terms, prob_n * params.risk_appetite * budget[n])
    end

    @objective(model, Max, obj_terms)

    verbose && println("  └─ ✅ Stochastic model construction complete!")
    verbose && println("\n📊 Stochastic Model Statistics:")
    verbose && println("    • Players: $(length(J))")
    verbose && println("    • Nodes: $(length(N))")
    verbose && println("    • Leaf nodes: $(length(data.leaf_nodes))")
    verbose && println("    • Chemistry Pairs: $(length(pairs))")
    verbose && println("    • Variables: $(num_variables(model))")
    verbose && println("    • Constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

    return model
end

# =============================================================================
# SOLVING
# =============================================================================

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

function solve_stochastic_model(
    model::Model,
    data::ModelDataStochastic,
    params::ModelParameters;
    verbose::Bool=true
)
    verbose && println("\n🚀 Solving stochastic node-based optimization model...")
    verbose && println("="^60)

    start_time = time()
    optimize!(model)
    solve_time = time() - start_time

    status = termination_status(model)
    verbose && println("\n📌 Termination Status: $status")

    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT]
        error("Stochastic model did not solve successfully. Status: $status")
    end

    obj_value = objective_value(model)
    verbose && println("🎯 Objective Value: $(round(obj_value, digits=2))")
    verbose && println("⏱️  Solve Time: $(round(solve_time, digits=2))s")
    verbose && println("\n📦 Extracting stochastic solution...")

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
        obj_value,
        solve_time,
        df_decisions,
        df_budget,
        df_tree,
        df_formation_diagnostics
    )
end

# =============================================================================
# EXPORT
# =============================================================================

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
