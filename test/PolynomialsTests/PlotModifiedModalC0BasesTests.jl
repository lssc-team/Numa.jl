using Gridap
# using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials

using Makie
using AbstractPlotting.MakieLayout

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (1200, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

T = Float64
maxod = 4
order = [1,2,3,4]
od = Node{Int64}(order[2])
menu = LMenu(scene, options = zip(["q = 1","q = 2","q = 3","q = 4"], order))
menu.direction = :up

ξ = collect(0.0:0.25:1.0)
x = collect(0.0:0.02:1.0)
xp = [Fields.Point(i) for i in x]

CIs = CartesianIndices((1:length(ξ),1:length(ξ)))

s1 = LSlider(scene,range=1:length(ξ),startvalue=1,
             color_inactive=RGBf0(1.0,1.0,1.0))
s2 = LSlider(scene,range=1:length(ξ),startvalue=1,
             color_inactive=RGBf0(1.0,1.0,1.0))

axM = layout[1, 1] = LAxis(scene)
axL = layout[1, 2] = LAxis(scene)
cols = [:red, :blue, :green, :orange, :purple]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot]

dataM = zeros(T,length(ξ),length(ξ),length(x),maxod+1)
dataL = zeros(T,length(ξ),length(ξ),length(x),maxod+1,maxod)

for ci in CIs

  ci[1] ≥ ci[2] && continue
  ξ₀=Fields.Point(ξ[ci[1]])
  ξ₁=Fields.Point(ξ[ci[2]])

  b = ModifiedModalC0Basis{1}(T,maxod,ξ₀=ξ₀,ξ₁=ξ₁)
  dataM[ci,:,:] = evaluate(b,xp)

  domain = (ξ[ci[1]],ξ[ci[2]])
  partition = (1,)
  model = CartesianDiscreteModel(domain,partition)
  for o in 1:maxod
    fe_cell = FiniteElements(PhysicalDomain(),model,:Lagrangian,T,o)
    V = FESpace(model,fe_cell)
    dv = get_cell_shapefuns(V)
    dataL[ci,:,1:o+1,o] = evaluate(dv.cell_basis[1],xp)
  end

end

for b in 1:maxod+1

  dataMb = lift(s1.value,s2.value,od) do iξ₀,iξ₁,odval
    if b ≤ odval+1
      x,dataM[iξ₀,iξ₁,:,b]
    else
      ([0.0,0.0],[0.0,0.0])
    end
  end
  lines!(axM,dataMb,color=cols[b],linestyle=lins[b],linewidth=2)

  dataLb = lift(s1.value,s2.value,od) do iξ₀,iξ₁,odval
    if b ≤ odval+1
      x,dataL[iξ₀,iξ₁,:,b,odval]
    else
      ([0.0,0.0],[0.0,0.0])
    end
  end
  lines!(axL,dataLb,color=cols[b],linestyle=lins[b],linewidth=2)

end

dataI = ([0.0,1.0],[0.0,0.0])
lines!(axM,dataI,color=:gray,linewidth=1.0)
lines!(axL,dataI,color=:gray,linewidth=1.0)
scatter!(axM,dataI,markersize=6,marker=:+,strokewidth=0.0)
scatter!(axL,dataI,markersize=6,marker=:+,strokewidth=0.0)
dataI = lift(s1.value,s2.value) do iξ₀,iξ₁
    ([ξ[iξ₀],ξ[iξ₁]],[0.0,0.0])
end
lines!(axM,dataI,color=:black,linewidth=1.5)
lines!(axL,dataI,color=:black,linewidth=1.5)
scatter!(axM,dataI,markersize=6)
scatter!(axL,dataI,markersize=6)

line1 = lines!(axM,(0,0),color=cols[1],linestyle=lins[1],linewidth=2)
line2 = lines!(axM,(0,0),color=cols[2],linestyle=lins[2],linewidth=2)
line3 = lines!(axM,(0,0),color=cols[3],linestyle=lins[3],linewidth=2)
line4 = lines!(axM,(0,0),color=cols[4],linestyle=lins[4],linewidth=2)
line5 = lines!(axM,(0,0),color=cols[5],linestyle=lins[5],linewidth=2)

axM.xticks = 0.0:0.1:1.0
# ylims!(axM,-37,37)
# axM.yticks = -35:10:35
axM.xlabel="ξ"
axM.ylabel="Modal φ(ξ)"

axL.xticks = 0.0:0.1:1.0
# ylims!(axL,-37,37)
# axL.yticks = -35:10:35
axL.xlabel="ξ"
axL.ylabel="Lagrangian φ(ξ)"

legmarkers = [ line1, line2, line3, line4, line5 ]
legnames = [ "φ₁", "φ₂", "φ₃", "φ₄", "φ₅" ]
leg = LLegend( scene, legmarkers, legnames )
error = (x,y) -> x < y ? " " : "ERROR! ξ₁ ≥ ξ₀"
layout[2, :] = hbox!( vbox!( LText(scene, "Order", width = nothing),
      menu, tellheight = false, width = 200 ), vbox!(
      LText(scene, lift(x -> "ξ₀: $(ξ[x])",s1.value), width = nothing), s1,
      LText(scene, lift(x -> "ξ₁: $(ξ[x])",s2.value), width = nothing), s2 ),
      LText(scene, lift(error,s1.value,s2.value), width = 150), leg )

on(s1.value) do s
  autolimits!(axM)
  autolimits!(axL)
end

on(s2.value) do s
  autolimits!(axM)
  autolimits!(axL)
end

on(menu.selection) do s
  od[] = s
  autolimits!(axM)
  autolimits!(axL)
end

scene
