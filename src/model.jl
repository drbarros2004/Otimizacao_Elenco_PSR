"""
Model entrypoint module.

This file intentionally stays small and only wires the split model files.
Keep main.jl compatibility via include("src/model.jl").
"""

include("model_base.jl")
include("solver_engine.jl")
include("result_exporter.jl")
include("model_deterministic.jl")
include("model_stochastic.jl")
