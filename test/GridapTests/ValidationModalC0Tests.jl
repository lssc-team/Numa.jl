using Test
using Gridap
import Gridap: ∇

const k = 2*pi
u(x) = sin(k*x[1]) * x[2]
∇u(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]))
# ∇u(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]), 0)
f(x) = (k^2)*sin(k*x[1])*x[2]

∇(::typeof(u)) = ∇u
b(v) = ∫( v*f )*dΩ

function run(n,order)

  domain = (0,1,0,1)
  partition = (n,n)
  # domain = (0,1,0,1,0,1)
  # partition = (n,n,n)
  model = CartesianDiscreteModel(domain,partition)

  reffe = ReferenceFE(:ModalC0,Float64,order)
  V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  U = TrialFESpace(V0,u)

  degree = 2*order
  Ω = Triangulation(model)
  dΩ = LebesgueMeasure(Ω,degree)

  a(u,v) = ∫( ∇(v)⊙∇(u) )*dΩ
  b(v) = ∫( v*f )*dΩ

  op = AffineFEOperator(a,b,U,V0)

  uh = solve(op)

  e = u - uh

  el2 = sqrt(sum( ∫( e*e )*dΩ ))
  eh1 = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))

  (el2, eh1)

end

function conv_test(ns,order)

  el2s = Float64[]
  eh1s = Float64[]
  hs = Float64[]

  for n in ns

    el2, eh1 = run(n,order)
    h = 1.0/n

    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(hs,h)

  end

  (el2s, eh1s, hs)

end

# using Plots

# plot(hs,[el2s eh1s],
#     xaxis=:log, yaxis=:log,
#     label=["L2" "H1"],
#     shape=:auto,
#     xlabel="h",ylabel="error norm")

# #src savefig("conv.png")

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

order = 1
el2s, eh1s, hs = conv_test([8,16,32,64,128],order)
@test slope(hs,el2s) > order+0.95
@test slope(hs,eh1s) > order-0.05

order = 2
el2s, eh1s, hs = conv_test([8,16,32,64,128],order)
@test slope(hs,el2s) > order+0.95
@test slope(hs,eh1s) > order-0.05

order = 3
el2s, eh1s, hs = conv_test([8,16,32,64],order)
@test slope(hs,el2s) > order+0.95
@test slope(hs,eh1s) > order-0.05

order = 4
el2s, eh1s, hs = conv_test([8,16,32],order)
@test slope(hs,el2s) > order+0.95
@test slope(hs,eh1s) > order-0.05

# # DIM 3

# order = 1
# el2s, eh1s, hs = conv_test([4,8,16],order)
# @test slope(hs,el2s) > order+0.95
# @test slope(hs,eh1s) > order-0.05

# order = 2
# el2s, eh1s, hs = conv_test([4,8,16],order)
# @test slope(hs,el2s) > order+0.95
# @test slope(hs,eh1s) > order-0.05
