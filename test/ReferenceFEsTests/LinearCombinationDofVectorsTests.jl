module LinearCombinationDofVectorsTests

using Test
using Gridap.Helpers
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Arrays
using Gridap.Polynomials
using Gridap.ReferenceFEs

# using BenchmarkTools
# import Gridap.ReferenceFEs: compute_nodes
# import Gridap.ReferenceFEs: return_cache, evaluate!

order = 1
test_lincom_dofvecs(Float64,SEGMENT,order)

order = 3
test_lincom_dofvecs(Float64,SEGMENT,order)

orders = (2,3)
test_lincom_dofvecs(Float64,QUAD,orders)

order = 2
test_lincom_dofvecs(VectorValue{2,Float64},QUAD,order)

order = 1
test_lincom_dofvecs(VectorValue{3,Float64},HEX,order)

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
