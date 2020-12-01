module ModalC0RefFEsTests

using Test
using Gridap
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.Fields

# using BenchmarkTools

# # Degenerated case
# order = 0
# reffe = ReferenceFE(QUAD,:ModalC0,Float64,order)

# # Error if create on simplices
# order = 1
# reffe = ReferenceFE(TRI,:ModalC0,Float64,order)

order = 1
p = QUAD
T = VectorValue{2,Float64}

m = ReferenceFE(p,:ModalC0,T,order)
l = ReferenceFE(p,:Lagrangian,T,order)

test_reference_fe(m)

@test num_dofs(m) == num_dofs(l)
@test Conformity(m) === Conformity(l)
@test get_face_own_dofs(m,Conformity(m)) == get_face_own_dofs(l,Conformity(l))
@test get_face_own_dofs_permutations(m,Conformity(m)) == get_face_own_dofs_permutations(l,Conformity(l))

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

function test_function_interpolation(::Type{T},order,C,u) where T
  reffe = ReferenceFE(:ModalC0,T,order)
  V = FESpace(model,reffe,conformity=C)
  test_single_field_fe_space(V)
  uh = interpolate(u,V)
  Ω = Triangulation(model)
  degree = 2*order
  dΩ = LebesgueMeasure(Ω,degree)
  l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
  e = u - uh
  el2 = l2(e)
  @test el2 < 1.0e-9
end

order = 2; T = Float64; C = :H1; u(x) = (x[1]+x[2])^2
test_function_interpolation(T,order,C,u)

order = 1; T = Float64; C = :L2; u(x) = x[1]+x[2]
test_function_interpolation(T,order,C,u)

order = 1; T = VectorValue{2,Float64}; C = :H1
u(x) = VectorValue(x[1]+x[2],x[2])
test_function_interpolation(T,order,C,u)

domain = (0,1,0,1,0,1)
partition = (2,2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1; T = Float64; C = :H1; u(x) = x[1]+x[2]+x[3]
test_function_interpolation(T,order,C,u)

# Inspect operator matrix to check if L2-scalar product of
# gradients of bubble functions satisfy Kronecker's delta
# domain = (0,1)
# partition = (1)
# model = CartesianDiscreteModel(domain,partition)
# order = 6; T = Float64; C = :H1;
# reffe = ReferenceFE(:ModalC0,T,order)
# V = FESpace(model,reffe,conformity=C)
# Ω = Triangulation(model)
# degree = 2*order
# dΩ = LebesgueMeasure(Ω,degree)
# a(u,v) = ∫( ∇(v)⊙∇(u) )*dΩ
# b(v) = 0.0
# op = AffineFEOperator(a,b,V,V)

end # module
