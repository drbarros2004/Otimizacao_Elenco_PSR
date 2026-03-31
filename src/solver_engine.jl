"""
Solver engine utilities.
Centralizes optimizer setup, solve lifecycle, status checks, and conflict diagnostics.
"""

using JuMP
const MOI = JuMP.MOI

struct SolverRunSummary
    status
    primal_status
    dual_status
    solve_time::Float64
    objective_value::Float64
end

"""
    configure_solver!(model; ...)

Applies standard solver parameters and optional numerical robustness settings.
"""
function configure_solver!(
    model::Model;
    time_limit::Float64 = 600.0,
    mip_gap::Float64 = 0.01,
    verbose::Bool = true,
    numeric_focus::Union{Nothing,Int} = nothing,
    scale_flag::Union{Nothing,Int} = nothing
)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "MIPGap", mip_gap)

    if !verbose
        set_optimizer_attribute(model, "OutputFlag", 0)
    end

    if !isnothing(numeric_focus)
        set_optimizer_attribute(model, "NumericFocus", numeric_focus)
    end

    if !isnothing(scale_flag)
        set_optimizer_attribute(model, "ScaleFlag", scale_flag)
    end

    return model
end

"""
    try_compute_conflict!(model; verbose=true)

Attempts to compute an IIS/conflict when the model is infeasible.
"""
function try_compute_conflict!(model::Model; verbose::Bool=true)
    backend_model = backend(model)

    try
        MOI.compute_conflict!(backend_model)
        cstatus = MOI.get(backend_model, MOI.ConflictStatus())
        if verbose
            println("🧪 Conflict/IIS status: $cstatus")
        end
        return cstatus
    catch e
        if verbose
            @warn "Unable to compute IIS/conflict diagnostics." exception = (e, catch_backtrace())
        end
        return nothing
    end
end

"""
    run_solver_with_status!(model; ...)

Executes optimize!, validates statuses, and returns a normalized solve summary.
"""
function run_solver_with_status!(
    model::Model;
    model_label::String = "optimization model",
    verbose::Bool = true,
    allow_time_limit::Bool = true,
    diagnose_conflict::Bool = true
)
    verbose && println("\n🚀 Solving $model_label...")
    verbose && println("="^60)

    start_time = time()
    optimize!(model)
    solve_time = time() - start_time

    status = termination_status(model)
    p_status = primal_status(model)
    d_status = dual_status(model)

    verbose && println("\n📌 Termination Status: $status")
    verbose && println("📌 Primal Status: $p_status")
    verbose && println("📌 Dual Status: $d_status")

    accepted_statuses = allow_time_limit ?
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT) :
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

    if status ∉ accepted_statuses
        if diagnose_conflict && status in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            try_compute_conflict!(model; verbose=verbose)
        end
        error("$model_label did not solve successfully. termination_status=$status")
    end

    feasible_primal = p_status in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    if !feasible_primal
        if diagnose_conflict && status in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
            try_compute_conflict!(model; verbose=verbose)
        end
        error("$model_label finished without a feasible primal point. primal_status=$p_status")
    end

    obj_value = objective_value(model)

    verbose && println("🎯 Objective Value: $(round(obj_value, digits=2))")
    verbose && println("⏱️  Solve Time: $(round(solve_time, digits=2))s")

    return SolverRunSummary(status, p_status, d_status, solve_time, obj_value)
end
