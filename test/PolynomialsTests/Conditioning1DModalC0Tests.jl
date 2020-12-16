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

function run(order::Int,ξ₁::Real)

  domain = (0,1)
  partition = (1,)
  model = CartesianDiscreteModel(domain,partition)

  degree = 2*(maxod-1)
  Ω = Triangulation(model)
  dΩ = LebesgueMeasure(Ω,degree)

  pt = Fields.Point(ξ₁)
  V = TestFESpace(model,ReferenceFE(:ModalC0,Float64,order,ξ₁=pt),dirichlet_tags=[1])
  U = TrialFESpace(V,u)

  a(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ
  l(v) = ∫( f*v )*dΩ

  op = AffineFEOperator(a,l,U,V)
  kop = cond(Array(get_matrix(op)))

  num_free_dofs(U),kop

end

function cond_test(np,ξ₁)

  ps = Float64[]
  ks = Float64[]

  for p in 1:np

    N, k = run(p,ξ₁)

    push!(ps,N)
    push!(ks,k)

  end

  (ps, ks)

end

ξ₁ = collect(0.001:0.001:1.0)
raw_data = [ cond_test(maxod,pξ₁) for pξ₁ in ξ₁ ]

function slope((hs,ks))
  x = log10.(hs)
  y = log10.(ks)
  linreg = hcat( fill!( similar(x), 1 ), x ) \ y
  linreg[2]
end

data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (600, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

ax = layout[1,1] = LAxis(scene)
cols = [:red, :blue, :green, :orange, :purple]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot]

line1 = lines!(ax,data,color=cols[1],linestyle=lins[1],linewidth=2)
limits!(ax, -0.05, 1.05, -1, 11)
ax.xticks = 0.0:0.25:1.0
ax.yticks = 0.0:2.0:10.0
ax.xlabel="ξ₁"
ax.ylabel="r"
ax.title="k(A) = O(Nʳ) where r = r(ξ₁)"

save("conds1DModalC0.png", scene)
