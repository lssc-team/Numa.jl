module LinearCombinationDofVectorsTests

using Test
using Gridap.Helpers
using Gridap.TensorValues
using Gridap.ReferenceFEs

using BenchmarkTools
import LinearAlgebra: I
import Gridap.Polynomials: ModalC0Basis
import Gridap.ReferenceFEs: compute_nodes
import Gridap.ReferenceFEs: evaluate

function test_lincom_dofvecs(::Type{T},p::Polytope{D},orders::NTuple{D,Int}) where {D,T}
  mb = ModalC0Basis{D}(T,orders)
  nodes, _ = compute_nodes(p,mb.orders)
  predofs = LagrangianDofBasis(T,nodes)
  change = inv(evaluate(predofs,mb))
  lincom_dofvals = linear_combination(change,predofs)
  id = Matrix{eltype(T)}(I,size(mb)[1],size(mb)[1])
  test_dof_array(lincom_dofvals,mb,id,cmp=(≈))
end

function test_lincom_dofvecs(::Type{T},p::Polytope{D},order::Int) where {D,T}
  orders = tfill(order,Val{D}())
  test_lincom_dofvecs(T,p,orders)
end

order = 1
test_lincom_dofvecs(Float64,SEGMENT,order)

# order = 3
# test_lincom_dofvecs(Float64,SEGMENT,order)

orders = (2,3)
test_lincom_dofvecs(Float64,QUAD,orders)

order = 1
test_lincom_dofvecs(VectorValue{2,Float64},QUAD,order)

# order = 1
# test_lincom_dofvecs(VectorValue{3,Float64},HEX,order)

# D = 2
# T = Float64
# orders = (2,3)
# mb = ModalC0Basis{D}(T,orders)
# nodes, _ = compute_nodes(QUAD,mb.orders)
# predofs = LagrangianDofBasis(T,nodes)
# change = inv(evaluate(predofs,mb))
# lincom_dofvals = linear_combination(change,predofs)
# cache = return_cache(lincom_dofvals,mb)
# @btime evaluate!($cache,$lincom_dofvals,$mb)

end # module
