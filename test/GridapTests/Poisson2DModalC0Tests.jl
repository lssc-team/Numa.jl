using Gridap
using Gridap.Fields
using LinearAlgebra: cond

using Makie
using AbstractPlotting.MakieLayout

const h = 1 # cell size
const maxod = 12

u(x) = -x[1]^maxod + maxod*x[1] -x[2]^maxod + maxod*x[2]
# u(x) = -(1/64)*sin(8*x[1])+(1/8)*cos(8)*x[1]
∇u(x) = ∇(u)(x)
f(x) = -Δ(u)(x)

function run(order::Int,ξ₁::Real,η₁::Real)

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
  # V = TestFESpace(model,ReferenceFE(:ModalC0,Float64,order,type=:modified,ξ₁=pt))
  # U = TrialFESpace(V)
  V = TestFESpace(model,ReferenceFE(:ModalC0,Float64,order,type=:modified,ξ₁=pt),dirichlet_tags=[1,2,3,5,7])
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

  uh = solve(op)

  # writevtk(Ω,"results",nsubcells=24,cellfields=["uh"=>uh])

  e = u - uh
  l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
  h1(u) = sqrt(sum( ∫( ∇(u)⊙∇(u) )*dΩ ))
  el2 = l2(e)
  eh1 = h1(e)

  sqrt(num_free_dofs(U)),el2,eh1,kop

end

function conv_test(np::Int,ξ₁::Real,η₁::Real)

  ps = Float64[]
  el2s = Float64[]
  eh1s = Float64[]
  ks = Float64[]

  for p in 1:np

    N, el2, eh1, k = run(p,ξ₁,η₁)

    push!(ps,N)
    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(ks,k)

  end

  (ps, el2s, eh1s, ks)

end

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (900, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

axE = layout[1,1] = LAxis(scene)
axC = layout[1,2] = LAxis(scene)
cols = [:red, :blue, :green, :orange, :purple]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot]

ξ₁ = collect(0.05:0.025:1.0)
s = LSlider(scene,range=1:length(ξ₁),startvalue=1,
            color_inactive=RGBf0(1.0,1.0,1.0))

data = zeros(length(ξ₁),maxod,4)
for (iξ₁,pξ₁) in enumerate(ξ₁)
  ps, el2s, eh1s, ks = conv_test(maxod,pξ₁,pξ₁)
  data[iξ₁,:,:] = hcat(ps,el2s,eh1s,ks)
end

datal2 = lift(s.value) do iξ₁
  log10.(data[iξ₁,:,1]),log10.(data[iξ₁,:,2])
end

datah1 = lift(s.value) do iξ₁
  log10.(data[iξ₁,:,1]),log10.(data[iξ₁,:,3])
end

datacn = lift(s.value) do iξ₁
  log10.(data[iξ₁,:,1]),log10.(data[iξ₁,:,4])
end

line1 = lines!(axE,datal2,color=cols[1],linestyle=lins[1],linewidth=2)
line2 = lines!(axE,datah1,color=cols[2],linestyle=lins[2],linewidth=2)
line3 = lines!(axC,datacn,color=cols[3],linestyle=lins[3],linewidth=2)

limits!(axE, -0.15, 1.15, -16, 1)
axE.xticks = 0.0:0.25:1.0
axE.yticks = -15.0:3.0:0.0
axE.xlabel="log10(N)"
axE.ylabel="log10(Abs error)"

limits!(axC, -0.15, 1.15, -1, 15)
axC.xticks = 0.0:0.25:1.0
axC.yticks = 0.0:2.0:14.0
axC.xlabel="log10(sqrt(N))"
axC.ylabel="log10(Condition number)"

axE.xticksize = 2.0; axE.yticksize = 2.0
axE.xticklabelsize = 12.0; axE.yticklabelsize = 12.0

axC.xticksize = 2.0; axC.yticksize = 2.0
axC.xticklabelsize = 12.0; axC.yticklabelsize = 12.0

legmarkers = [ line1, line2, line3 ]
legnames = [ "L2", "H1", "κ(A)"]
leg = LLegend( scene, legmarkers, legnames, orientation = :horizontal )
layout[2, :] =
  hbox!( LText(scene, lift(x -> "ξ₁: $(ξ₁[x])",s.value), width = 60), s, leg )

# record(scene,"poisson2DModalC0.mp4",length(ξ₁):-1:1; framerate = 1) do i
#   s.value = i
# end
scene
