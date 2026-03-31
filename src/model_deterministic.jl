"""
Deterministic Squad Optimization Model (Multi-Period MILP).
"""

function build_squad_optimization_model(data::ModelData, params::ModelParameters; verbose::Bool=true)
    verbose && println("\n Building Complete Multi-Period Squad Optimization Model...")

    model = Model(Gurobi.Optimizer)
    configure_solver!(model; verbose=verbose)

    # Extract data
    players = data.players
    J = players.player_id
    T = data.windows
    pos_groups = Dict(row.player_id => row.pos_group for row in eachrow(players))
    formation_positions = String[]
    for limits in values(params.formation_catalog)
        append!(formation_positions, collect(keys(limits)))
    end
    formation_positions = unique(sort(formation_positions))

    if isempty(formation_positions)
        error("No tactical positions found in formation catalog.")
    end

    # Generate pairs for chemistry
    pairs = generate_player_pairs(data.initial_squad)
    verbose && println("  ├─ Chemistry pairs: $(length(pairs))")

    # ==== DECISION VARIABLES ====
    verbose && println("  ├─ Creating decision variables...")

    @variable(model, x[j in J, t in T], Bin)
    @variable(model, y[j in J, t in T], Bin)
    @variable(model, buy[j in J, t in T], Bin)
    @variable(model, sell[j in J, t in T], Bin)
    @variable(model, budget[t in T] >= 0)
    @variable(model, budget_deficit[t in T] >= 0)

    # Chemistry variables
    @variable(model, ParElenco[(i,j) in pairs, t in T], Bin)
    @variable(model, ParTitular[(i,j) in pairs, t in T], Bin)
    @variable(model, 0 <= Quimica[(i,j) in pairs, t in T] <= S_MAX)

    # Soft constraint slacks
    @variable(model, slack_posicao[p in formation_positions, t in T] >= 0)
    @variable(model, slack_salario[t in T] >= 0)

    # ==== INITIAL CONDITIONS ====
    verbose && println("  ├─ Initial conditions...")

    @constraint(model, budget[first(T)] == params.initial_budget)

    for j in J
        if j in data.initial_squad
            @constraint(model, x[j, first(T)] == 1)
            @constraint(model, buy[j, first(T)] == 0)
            @constraint(model, sell[j, first(T)] == 0)
        else
            @constraint(model, x[j, first(T)] == 0)
            @constraint(model, buy[j, first(T)] == 0)  # Cannot buy in initial window
            @constraint(model, sell[j, first(T)] == 0) # Cannot sell what you don't have
        end
    end

    # Initial chemistry for base pairs
    for (i, j) in pairs
        if i in data.initial_squad && j in data.initial_squad
            @constraint(model, Quimica[(i,j), first(T)] == S_INICIAL)
        else
            @constraint(model, Quimica[(i,j), first(T)] == 0)
        end
    end

    # ==== SQUAD FLOW (timing: buy at t-1 affects squad at t) ====
    verbose && println("  ├─ Squad flow constraints (Python timing)...")

    for j in J, t in T
        if t > first(T)
            @constraint(model, x[j, t] == x[j, t-1] + buy[j, t-1] - sell[j, t-1])
        end
    end

    # ==== BUDGET DYNAMICS ====
    verbose && println("  ├─ Budget constraints...")

    for t in T
        if t > first(T)
            revenue = (t % 2 == 0) ? params.seasonal_revenue : 0.0

            # Calculate signing costs with free agent multiplier
            # Free agents (cost < 100k) demand higher signing bonuses
            signing_costs = AffExpr(0.0)
            for j in J
                cost = data.cost_map[(j, t)]
                wage = data.wage_map[(j, t)]
                ovr = data.ovr_map[(j, t)]

                # Determine if player is a free agent
                is_free_agent = (cost < 100_000)

                if is_free_agent
                    # Apply OVR-based multiplier for free agents
                    multiplier = get_free_agent_signing_multiplier(ovr)
                    signing_cost = wage * 52 * params.signing_bonus_rate * multiplier
                else
                    # Normal signing bonus for regular transfers
                    signing_cost = wage * 52 * params.signing_bonus_rate
                end

                add_to_expression!(signing_costs, signing_cost * buy[j, t-1])
            end

            @constraint(model,
                budget[t] == budget[t-1]
                    - sum((1 + params.transaction_cost_buy) * data.cost_map[(j,t)] * buy[j,t-1] for j in J)
                    - signing_costs
                    + sum((1 - params.transaction_cost_sell) * data.value_map[(j,t)] * sell[j,t-1] for j in J)
                    + revenue
                    + budget_deficit[t]
            )
        end
    end

    # ==== SQUAD SIZE ====
    verbose && println("  ├─ Squad size constraints...")

    for t in T
        @constraint(model, sum(x[j,t] for j in J) <= params.max_squad_size)
        @constraint(model, sum(x[j,t] for j in J) >= params.min_squad_size)
    end

    # ==== SALARY CAP (SOFT) ====
    verbose && println("  ├─ Salary cap constraints (soft)...")

    # Baseline payroll comes from the initial squad at window 0.
    initial_window = first(T)
    initial_payroll = sum(data.wage_map[(j, initial_window)] for j in data.initial_squad)
    salary_cap_per_window = initial_payroll * params.salary_cap_multiplier_initial * params.salary_cap_window_factor

    verbose && println("  │  Initial payroll baseline: €$(round(initial_payroll / 1e6, digits=2))M")
    verbose && println("  │  Salary cap per window: €$(round(salary_cap_per_window / 1e6, digits=2))M")

    for t in T
        @constraint(model,
            sum(data.wage_map[(j,t)] * x[j,t] for j in J) <= salary_cap_per_window + slack_salario[t]
        )
    end

    # ==== FORMATION (EXACT) ====
    verbose && println("  ├─ Tactical formation constraints (exact)...")

    for t in T
        @constraint(model, sum(y[j,t] for j in J) == 11)

        _, formation_t = get_window_formation(params, t)

        for (pos, required_count) in formation_t
            players_in_pos = [j for j in J if pos_groups[j] == pos]
            @constraint(model,
                sum(y[j,t] for j in players_in_pos) == required_count
            )
        end
    end

    # ==== STARTER ELIGIBILITY ====
    verbose && println("  ├─ Starter eligibility...")

    for j in J, t in T
        @constraint(model, y[j,t] <= x[j,t])
    end

    # ==== TRANSACTION EXCLUSIVITY ====
    verbose && println("  ├─ Transaction exclusivity...")

    for j in J, t in T
        @constraint(model, buy[j,t] + sell[j,t] <= 1)
    end

    # ==== CHEMISTRY LINEARIZATION ====
    verbose && println("  ├─ Chemistry linearization...")

    for (i, j) in pairs, t in T
        # ParElenco = x[i] × x[j]
        @constraint(model, ParElenco[(i,j), t] <= x[i, t])
        @constraint(model, ParElenco[(i,j), t] <= x[j, t])
        @constraint(model, ParElenco[(i,j), t] >= x[i,t] + x[j,t] - 1)

        # ParTitular = y[i] × y[j]
        @constraint(model, ParTitular[(i,j), t] <= y[i, t])
        @constraint(model, ParTitular[(i,j), t] <= y[j, t])
        @constraint(model, ParTitular[(i,j), t] >= y[i,t] + y[j,t] - 1)
    end

    # ==== CHEMISTRY DYNAMICS ====
    verbose && println("  ├─ Chemistry temporal dynamics...")

    for (i, j) in pairs, t in T
        if t > first(T)
            M = S_MAX

            # Case 1: Together → chemistry grows
            @constraint(model,
                Quimica[(i,j), t] <= Quimica[(i,j), t-1] + BONUS_INCREMENTO + M * (1 - ParElenco[(i,j), t])
            )
            @constraint(model,
                Quimica[(i,j), t] >= Quimica[(i,j), t-1] + BONUS_INCREMENTO - M * (1 - ParElenco[(i,j), t])
            )

            # Case 2: Separated → chemistry decays
            @constraint(model,
                Quimica[(i,j), t] <= params.decay_quimica * Quimica[(i,j), t-1] + M * ParElenco[(i,j), t]
            )
            @constraint(model,
                Quimica[(i,j), t] >= params.decay_quimica * Quimica[(i,j), t-1] - M * ParElenco[(i,j), t]
            )
        end
    end

    # ==== OBJECTIVE FUNCTION ====
    verbose && println("  ├─ Building objective function...")

    obj_terms = AffExpr(0.0)

    # 1. Score de Mercado (Asset Value) - applies to ALL squad
    for j in J, t in T
        if t > first(T)  # Skip t=0
            score_mercado = (
                params.weight_quality * data.ovr_map[(j,t)] +
                params.weight_potential * data.growth_potential_map[(j,t)]
            )
            add_to_expression!(obj_terms, params.peso_asset * score_mercado * x[j,t])
        end
    end

    # 2. Score Tático (Performance) - applies to STARTERS
    for j in J, t in T
        if t > first(T)
            score_tatico = data.ovr_map[(j,t)] * 1.0
            add_to_expression!(obj_terms, params.peso_performance * score_tatico * y[j,t])
        end
    end

    # 3. Chemistry bonus
    for (i, j) in pairs, t in T
        if t > first(T)
            add_to_expression!(obj_terms, params.bonus_entrosamento * Quimica[(i,j), t])
        end
    end

    # 4. Friction penalty
    for j in J, t in T
        if t > first(T) && t > 1  # buy at t-1 affects t
            add_to_expression!(obj_terms, -params.friction_penalty * buy[j, t-1])
        end
    end

    # 5. Soft constraint penalties
    for t in T
        add_to_expression!(obj_terms, -params.salary_cap_penalty * slack_salario[t])
        add_to_expression!(obj_terms, -P_CAIXA * budget_deficit[t])
    end

    # 6. Terminal value
    t_final = last(T)
    valor_elenco_final = sum(data.value_map[(j, t_final)] * x[j, t_final] for j in J)
    add_to_expression!(obj_terms, 0.001 * valor_elenco_final)
    add_to_expression!(obj_terms, params.risk_appetite * budget[t_final])

    @objective(model, Max, obj_terms)

    verbose && println("  └─ ✅ Model construction complete!")
    verbose && println("\n📊 Model Statistics:")
    verbose && println("    • Players: $(length(J))")
    verbose && println("    • Windows: $(length(T))")
    verbose && println("    • Chemistry Pairs: $(length(pairs))")
    verbose && println("    • Variables: $(num_variables(model))")
    verbose && println("    • Constraints: $(num_constraints(model; count_variable_in_set_constraints=false))")

    return model
end

function solve_model(model::Model, data::ModelData, params::ModelParameters; verbose::Bool=true)
    summary = run_solver_with_status!(
        model;
        model_label="deterministic optimization model",
        verbose=verbose,
        allow_time_limit=true,
        diagnose_conflict=true
    )

    verbose && println("\n📦 Extracting solution...")

    return extract_deterministic_results(
        model,
        data;
        objective_value=summary.objective_value,
        solve_time=summary.solve_time,
        verbose=verbose
    )
end
