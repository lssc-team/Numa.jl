module JacobiPolynomialBasisTests

using Test
using Gridap.CellData
using Gridap.Integration
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials

using LinearAlgebra: I

order = 6
V = Float64
b = JacobiPolynomialBasis{1}(V,order)

quad = Quadrature(SEGMENT,2*order)
v = evaluate(b,quad.coordinates)

function compute_mass_matrix()
  M = zeros(Float64,length(b),length(b))
  for i = 1:length(quad.weights)
    M += quad.weights[i] * v[i,:] * transpose(v[i,:])
  end
  M
end

M = compute_mass_matrix()
id = Matrix{Float64}(I,length(b),length(b))
@test M ≈ id

end # module
