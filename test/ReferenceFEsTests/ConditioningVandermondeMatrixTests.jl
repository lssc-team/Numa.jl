using Gridap
using Gridap.Polynomials
using Gridap.ReferenceFEs
using LinearAlgebra: cond

using Makie
using AbstractPlotting.MakieLayout

function run(order::Int)

  prebasis = MonomialBasis(Float64,SEGMENT,order)
  nodes, _ = compute_nodes(SEGMENT,prebasis.orders)
  predofs = LagrangianDofBasis(Float64,nodes)
  change = evaluate(predofs,prebasis)
  kop = cond(change)
  log10(kop)

end

function cond_test(np)

  ps = Float64[]
  ks = Float64[]

  for p in 1:np

    push!(ps,p)
    push!(ks,run(p))

  end

  (ps, ks)

end

# ξ₁ = collect(0.001:0.001:1.0)
# raw_data = [ cond_test(maxod,pξ₁) for pξ₁ in ξ₁ ]

# function slope((hs,ks))
#   x = log10.(hs)
#   y = log10.(ks)
#   linreg = hcat( fill!( similar(x), 1 ), x ) \ y
#   linreg[2]
# end

# data = [ (pξ₁,slope(raw_data[iξ₁])) for (iξ₁,pξ₁) in enumerate(ξ₁) ]
data = cond_test(12)

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (600, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

ax = layout[1,1] = LAxis(scene)
cols = [:red, :blue, :green, :orange, :purple]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot]

line1 = lines!(ax,data,color=cols[1],linestyle=lins[1],linewidth=2)
limits!(ax, 0.5, 12.5, -1, 11)
ax.xticks = 1.0:1.0:12.0
ax.yticks = 0.0:1.0:10.0

ax.xlabel="order"
ax.ylabel="κ"
ax.title="κ(V) where V Vandermonde matrix mon-to-nod"

save("condsVandermondeMatrix.png", scene)
