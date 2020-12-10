using Gridap
# using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials

using Base.Iterators: product

using Makie
using AbstractPlotting.MakieLayout

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (700, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

T = Float64
order = 2
nshap = (order+1)*(order+1)

anchs = [0.0,0.25,0.5,0.75]
sides = [0.25,0.5,1.0]

x = product(0.0:0.05:1.0,0.0:0.05:1.0)
xp = reshape([Fields.Point(i) for i in x],length(x))
data = zeros(T,length(anchs),length(anchs),length(sides),length(x),nshap)

CIs = CartesianIndices((1:length(anchs),1:length(anchs),1:length(sides)))

for ci in CIs

  (((ci[1] > 1) | (ci[2] > 1)) & (ci[3] > 2)) && continue
  (((ci[1] > 3) | (ci[2] > 3)) & (ci[3] > 1)) && continue

  ξ₀ = Fields.Point(anchs[ci[1]],anchs[ci[2]])
  ss = Fields.Point(sides[ci[3]],sides[ci[3]])
  ξ₁ = ξ₀ + ss

  b = ModifiedModalC0Basis{2}(T,order,ξ₀=ξ₀,ξ₁=ξ₁)
  data[ci,:,:] = evaluate(b,xp)

end

s1 = LSlider(scene,range=1:length(anchs),startvalue=1,
             color_inactive=RGBf0(1.0,1.0,1.0),width=100) # ξ₀
s2 = LSlider(scene,range=1:length(anchs),startvalue=1,
             color_inactive=RGBf0(1.0,1.0,1.0),width=100) # η₀
s3 = LSlider(scene,range=1:length(sides),startvalue=3,
             color_inactive=RGBf0(1.0,1.0,1.0),width=100) # l

p1 = layout[3,1] = LAxis(scene) # φ₁
p2 = layout[3,3] = LAxis(scene) # φ₂
p3 = layout[1,1] = LAxis(scene) # φ₃
p4 = layout[1,3] = LAxis(scene) # φ₄
p5 = layout[3,2] = LAxis(scene) # φ₅
p6 = layout[1,2] = LAxis(scene) # φ₆
p7 = layout[2,1] = LAxis(scene) # φ₇
p8 = layout[2,3] = LAxis(scene) # φ₈
p9 = layout[2,2] = LAxis(scene) # φ₉
PP = (p1,p2,p3,p4,p5,p6,p7,p8,p9)

xi = collect(0.0:0.05:1.0)
upper = maximum(maximum(data[s1.value[],s2.value[],s3.value[],:,:],dims=1))
lower = minimum(minimum(data[s1.value[],s2.value[],s3.value[],:,:],dims=1))

D = []; H = []; C = []
for (i,p) in enumerate(PP)
  dataP = lift(s1.value,s2.value,s3.value) do iξ₀,iη₀,il
    reshape(data[iξ₀,iη₀,il,:,i],length(xi),length(xi))
  end
  heatP = heatmap!(p,xi,xi,dataP,interpolate=true)
  heatP.colorrange = (lower,upper)
  cellP = lift(s1.value,s2.value,s3.value) do iξ₀,iη₀,il
    [ (anchs[iξ₀],anchs[iη₀]),
      (anchs[iξ₀]+sides[il],anchs[iη₀]),
      (anchs[iξ₀]+sides[il],anchs[iη₀]+sides[il]),
      (anchs[iξ₀],anchs[iη₀]+sides[il]) ]
  end
  cellP = poly!(p,cellP,color=:transparent,strokecolor=:black,strokewidth=1.0)
  push!(D,dataP); push!(H,heatP); push!(C,cellP)
end

for p in PP
  p.xticksize = 2.0; p.yticksize = 2.0;
  p.xticklabelsize = 11.0; p.yticklabelsize = 11.0;
end

layout[4,1] = hbox!(
      LText(scene, lift(x -> "ξ₀: $(anchs[x])",s1.value), textsize = 14.0, width = 50, tellwidth = false), s1 )

layout[4,2] = hbox!(
      LText(scene, lift(x -> "η₀: $(anchs[x])",s2.value), textsize = 14.0, width = 50, tellwidth = false), s2 )

layout[4,3] = hbox!(
      LText(scene, lift(x -> "l: $(sides[x])",s3.value), textsize = 14.0, width = 50, tellwidth = false), s3 )

cbar = layout[:,4] = LColorbar(scene,H[1])
cbar.width = 15
cbar.height = Relative(2/3)
cbar.ticks = [round(x;digits=2) for x in LinRange(lower,upper,5)]
cbar.ticksize = 6.0
cbar.ticklabelsize = 14.0

function update_ranges()
  upper = maximum(maximum(data[s1.value[],s2.value[],s3.value[],:,:],dims=1))
  lower = minimum(minimum(data[s1.value[],s2.value[],s3.value[],:,:],dims=1))
  for h in H
    h.colorrange = (lower,upper)
  end
  cbar.ticks = [round(x;digits=2) for x in LinRange(lower,upper,5)]
end

on(s1.value) do s
  update_ranges()
end

on(s2.value) do s
  update_ranges()
end

on(s3.value) do s
  update_ranges()
end

scene
