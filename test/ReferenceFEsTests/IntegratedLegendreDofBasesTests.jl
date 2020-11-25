module IntegratedLegendreDofBasesTests

using Test
using Gridap.Helpers
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Arrays
using Gridap.Polynomials
using Gridap.ReferenceFEs
# using BenchmarkTools

import Gridap.Fields: Broadcasting
import Gridap.Arrays: return_cache, evaluate!
import Gridap.ReferenceFEs: Quadrature
import Gridap.ReferenceFEs: get_coordinates, get_weights
import Gridap.ReferenceFEs: compute_nodes, evaluate
import Gridap.ReferenceFEs: _evaluate_lagr_dof!

include("../../src/ReferenceFEs/IntegratedLegendreDofBases.jl")

# Scalar: 1D Linear
D = 2
V = Float64
order = (2,3)

b = IntegratedLegendreBasis{D}(V,order)
# _IL_nodes_and_moments!(QUAD,b)

lag_nodes, _ = compute_nodes(QUAD,b.orders)
lag_dofs = LagrangianDofBasis(V,lag_nodes)

change = inv(evaluate(lag_dofs,b))
lag_interpolation = evaluate(lag_dofs,b)
mod_interpolation = evaluate(*,transpose(lag_interpolation),transpose(change))

mod_dofs = IntegratedLegendreDofBasis(V,QUAD,b)
@test mod_interpolation ≈ evaluate(mod_dofs,b)

# # Scalar: 1D Linear
# D = 1
# order = 1
# V = Float64
# # b = IntegratedLegendreBasis{D}(V,order)
# # nodes, f_moments, f_nodes = _IL_nodes_and_moments(SEGMENT,b)
# dofs1 = IntegratedLegendreRefFE(V,SEGMENT,order)

# # Scalar: 1D Cubic
# D = 1
# order = 3
# V = Float64
# # b = IntegratedLegendreBasis{D}(V,order)
# # nodes2, f_moments2, f_nodes2 = _IL_nodes_and_moments(SEGMENT,b)
# dofs2 = IntegratedLegendreRefFE(V,SEGMENT,order)

# # Scalar: 2D x-Quad y-Cubic
# D = 2
# orders = (2,3)
# V = Float64
# # b = IntegratedLegendreBasis{D}(V,orders)
# # nodes3, f_moments3, f_nodes3, f_dofs3, vals = _IL_nodes_and_moments!(QUAD,b)
# dofs3 = IntegratedLegendreRefFE(V,QUAD,orders)

# # # Scalar: 3D Quadratic
# # D = 3
# # order = 2
# # V = Float64
# # b = IntegratedLegendreBasis{D}(V,order)
# # nodes4, f_moments4, f_nodes4 = _IL_nodes_and_moments(HEX,b)

# # cache = return_cache(b,[xi,])
# # @btime evaluate!($cache,$b,$[xi,])
# # cache = return_cache(∇b,[xi,])
# # @btime evaluate!($cache,$∇b,$[xi,])
# # cache = return_cache(∇∇b,[xi,])
# # @btime evaluate!($cache,$∇∇b,$[xi,])

end # module
