using Gridap
using Gridap.Fields
using LinearAlgebra: cond

using Makie
using AbstractPlotting.MakieLayout

const h = 1 # cell size
const maxod = 12

u(x) = -x[1]^maxod+maxod*x[1]
∇u(x) = ∇(u)(x)
f(x) = -Δ(u)(x)

function run(order::Int,β::Real,ξ₁::Real,η₁::Real)

  domain = (0,1,0,1)
  partition = (1,1,)
  model = CartesianDiscreteModel(domain,partition)

  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)

  degree = 2*maxod
  dΩ = LebesgueMeasure(Ω,degree)
  dΓ = LebesgueMeasure(Γ,degree)
  n_Γ = get_normal_vector(Γ)

  pt = Fields.Point(ξ₁,η₁)
  # V = TestFESpace(model,ReferenceFE(:ModalC0,Float64,order,ξ₁=pt))
  # U = TrialFESpace(V)
  V = TestFESpace(model,ReferenceFE(:ModalC0,Float64,order,ξ₁=pt),dirichlet_tags=[1,2,3,5,7])
  U = TrialFESpace(V,u)

  # γ = 2.5*order^2
  # a(u,v) =
  #   ∫( ∇(v)⋅∇(u) )*dΩ +
  #   ∫( (γ/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )*dΓ
  # l(v) =
  #   ∫( v*f )*dΩ +
  #   ∫( (γ/h)*v*u - (n_Γ⋅∇(v))*u )*dΓ

  a(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ
  l(v) = ∫( f*v )*dΩ

  op = AffineFEOperator(a,l,U,V)
  kop = cond(Array(get_matrix(op)))

  sqrt(num_free_dofs(U)),kop

end

function cond_test(np::Int,β::Real,ξ₁::Real,η₁::Real)

  ps = Float64[]
  ks = Float64[]

  for p in 1:np

    N, k = run(p,β,ξ₁,η₁)

    push!(ps,N)
    push!(ks,k)

  end

  (ps, ks)

end

function slope((hs,ks))
  x = log10.(hs)
  y = log10.(ks)
  linreg = hcat( fill!( similar(x), 1 ), x ) \ y
  linreg[2]
end

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (600, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

ax = layout[1,1] = LAxis(scene)
cols = [:red, :blue, :green, :orange, :purple]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot]

ξ₁ = collect(0.01:0.01:1.0)

raw_data = [ cond_test(maxod,10.0,pξ₁,pξ₁) for pξ₁ in ξ₁ ]
data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]
line1 = lines!(ax,data,color=cols[1],linestyle=lins[1],linewidth=2)

# raw_data = [ cond_test(maxod,5.0,pξ₁,pξ₁) for pξ₁ in ξ₁ ]
# data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]
# line2 = lines!(ax,data,color=cols[2],linestyle=lins[2],linewidth=2)

# raw_data = [ cond_test(maxod,2.5,pξ₁,pξ₁) for pξ₁ in ξ₁ ]
# data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]
# line3 = lines!(ax,data,color=cols[3],linestyle=lins[3],linewidth=2)

# raw_data = [ cond_test(maxod,1.0,pξ₁,pξ₁) for pξ₁ in ξ₁ ]
# data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]
# line4 = lines!(ax,data,color=cols[4],linestyle=lins[4],linewidth=2)

limits!(ax, -0.15, 1.15, -1, 17)
ax.xticks = 0.0:0.25:1.0
ax.yticks = 0.0:2.0:16.0
ax.xlabel="ξ₁"
ax.ylabel="r"
ax.title="k(A) = O(Nʳ) where r = r(ξ₁)"

# legmarkers = [ line1, line2, line3, line4 ]
# legnames = [ "β = 10.0", "β = 5.0", "β = 2.5", "β = 1.0"]
# layout[1,2] = LLegend( scene, legmarkers, legnames )

save("conds2DModalC0.png", scene)
