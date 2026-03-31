"""
Model entrypoint module.

This file intentionally stays small and only wires the split model files.
Keep main.jl compatibility via include("src/model.jl").
"""

include("model_base.jl")
include("model_deterministic.jl")
include("model_stochastic.jl")
