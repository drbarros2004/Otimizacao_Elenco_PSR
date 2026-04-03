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
    configure_solver!(model; verbose=verbose)

    # Extract data
    players = data.players
    J = Int.(players.player_id)
    N = sort(collect(keys(data.tree.nodes)))
    root_id = data.tree.root_id
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

    @constraint(model, budget[root_id] == money_to_millions(params.initial_budget))

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

        parent = data.tree.nodes[n].parent_id
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

        parent = data.tree.nodes[n].parent_id
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int

        stage_n = data.tree.nodes[n].stage
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

            add_to_expression!(signing_costs, money_to_millions(signing_cost) * buy[j, p])
        end

        @constraint(model,
            budget[n] == budget[p]
                - sum(money_to_millions((1 + params.transaction_cost_buy) * data.cost_node_map[(j,n)]) * buy[j,p] for j in J)
                - signing_costs
                + sum(money_to_millions((1 - params.transaction_cost_sell) * data.value_node_map[(j,n)]) * sell[j,p] for j in J)
                + money_to_millions(revenue)
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

    initial_payroll_million = sum(money_to_millions(data.wage_node_map[(j, root_id)]) for j in data.initial_squad)
    salary_cap_per_node = initial_payroll_million * params.salary_cap_multiplier_initial

    verbose && println("  │  Initial payroll baseline: €$(round(initial_payroll_million, digits=2))M")
    verbose && println("  │  Salary cap per node: €$(round(salary_cap_per_node, digits=2))M")

    for n in N
        @constraint(model,
            sum(money_to_millions(data.wage_node_map[(j,n)]) * x[j,n] for j in J) <= salary_cap_per_node + slack_salario[n]
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

    # ==== SQUAD DEPTH BY POSITION (SOFT UPPER BOUNDS) ====
    verbose && println("  ├─ Squad depth constraints by node (soft)...")

    for n in N
        for pos in formation_positions
            required_starters = get(data.position_requirements_map, (pos, n), 0)
            allowed_bench = get(params.bench_targets, pos, 0)
            max_allowed = required_starters + allowed_bench
            players_in_pos = [j for j in J if pos_groups[j] == pos]

            @constraint(model,
                sum(x[j,n] for j in players_in_pos) <= max_allowed + slack_posicao[pos,n]
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

        parent = data.tree.nodes[n].parent_id
        if isnothing(parent)
            error("Node $n has no parent and is not root node $(root_id).")
        end
        p = parent::Int

        @constraint(model,
            Quimica[(i,j), n] <= params.decay_quimica * Quimica[(i,j), p] 
                                + params.bonus_titular * ParTitular[(i,j), n] 
                                + params.bonus_elenco * (ParElenco[(i,j), n] - ParTitular[(i,j), n])
        )
    end

    # ==== EXPECTED-VALUE OBJECTIVE (NODE-INDEXED) ====
    verbose && println("  ├─ Building expected-value objective...")

    obj_terms = AffExpr(0.0)

    non_root_nodes = [n for n in N if n != root_id]

    for n in non_root_nodes
        prob_n = data.tree.nodes[n].cumulative_probability
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

        for p in formation_positions
            add_to_expression!(obj_terms, -prob_n * params.squad_position_penalty * slack_posicao[p,n])
        end

        add_to_expression!(obj_terms, -prob_n * params.salary_cap_penalty * slack_salario[n])
        add_to_expression!(obj_terms, -prob_n * P_CAIXA * budget_deficit[n])
    end

    for n in non_root_nodes
        prob_n = data.tree.nodes[n].cumulative_probability
        if prob_n <= 0.0
            continue
        end

        chemistry_multiplier = get(data.chemistry_multiplier_map, n, 1.0)
        for (i, j) in pairs
            add_to_expression!(obj_terms, prob_n * params.bonus_entrosamento * chemistry_multiplier * Quimica[(i,j), n])
        end
    end

    # Terminal value is only accounted for leaf nodes, weighted by cumulative probability.
    for n in data.tree.leaf_nodes
        prob_n = data.tree.nodes[n].cumulative_probability
        if prob_n <= 0.0
            continue
        end

        valor_elenco_final_million = sum(money_to_millions(data.value_node_map[(j, n)]) * x[j, n] for j in J)
        add_to_expression!(obj_terms, prob_n * 0.001 * valor_elenco_final_million)
        add_to_expression!(obj_terms, prob_n * params.risk_appetite * budget[n])
    end

    @objective(model, Max, obj_terms)

    verbose && println("  └─ ✅ Stochastic model construction complete!")
    verbose && println("\n📊 Stochastic Model Statistics:")
    verbose && println("    • Players: $(length(J))")
    verbose && println("    • Nodes: $(length(N))")
    verbose && println("    • Leaf nodes: $(length(data.tree.leaf_nodes))")
    verbose && println("    • Chemistry Pairs: $(length(pairs))")
    verbose && println("    • Variables: $(num_variables(model))")
    verbose && println("    • Constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

    return model
end

# =============================================================================
# SOLVING
# =============================================================================

function solve_stochastic_model(
    model::Model,
    data::ModelDataStochastic,
    params::ModelParameters;
    verbose::Bool=true
)
    summary = run_solver_with_status!(
        model;
        model_label="stochastic node-based optimization model",
        verbose=verbose,
        allow_time_limit=true,
        diagnose_conflict=true
    )

    verbose && println("\n📦 Extracting stochastic solution...")

    return extract_stochastic_results(
        model,
        data;
        objective_value=summary.objective_value,
        solve_time=summary.solve_time,
        verbose=verbose
    )
end
