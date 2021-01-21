using Gridap
using Gridap.Fields
using Gridap.CellData
using Gridap.Arrays
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.ReferenceFEs
using LinearAlgebra: cond
using Test

using Makie
using AbstractPlotting
using AbstractPlotting.MakieLayout

const maxod = 8
u(x) = -x[1]^maxod+maxod*x[1]
g(x) = ∇(u)(x)
f(x) = -Δ(u)(x)

hs(x::Real) = x > 1.0 ? 1.0 : 0.0

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (900, 600),
                            backgroundcolor = RGBf0(0.99, 0.99, 0.99))

function run_modal(od::Int,κ::Real)

  function stretching(x::Fields.Point)
    m = zeros(length(x))
    m[1] = x[1] + hs(x[1])*κ
    Fields.Point(m)
  end

  n = 2; partition = (n); domain = (0,2)
  model = CartesianDiscreteModel(domain,partition,map=stretching)

  bboxes = [ Fields.Point{1,Float64}[(0.0),(1.0),(0.0),(1.0),(0.0),(2.0+κ)],
             Fields.Point{1,Float64}[(0.0),(1.0),(0.0),(1.0),(0.0),(1.0)] ]

  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)

  degree = 2*od
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  reffe = ReferenceFE(modalC0,Float64,od,bboxes)
  Vstd = TestFESpace(model,reffe,dirichlet_tags=[1])

  aggdof_to_dof = vcat(2,collect(2+od:2*od))

  aggdof_to_dofs_ptrs = [1+i*(od+1) for i in 0:od]
  aggdof_to_dofs_data = repeat(vcat(-1,1,collect(3:od+1)),od)
  aggdof_to_dofs = Table(aggdof_to_dofs_data,aggdof_to_dofs_ptrs)

  sh1 = get_cell_shapefuns(Vstd).cell_basis[1]
  df2 = get_cell_dof_basis(Vstd).cell_dof[2]
  ld2 = ( ( 1.0 + κ ) .* df2.dof_basis.nodes ) .+ 1.0
  ld2 = LagrangianDofBasis(df2.dof_basis,ld2)
  df2to1 = linear_combination(df2,ld2)
  aggdof_to_coeffs_data = transpose(evaluate(df2to1,sh1))[od+2:end]
  aggdof_to_coeffs = Table(aggdof_to_coeffs_data,aggdof_to_dofs_ptrs)

  V = FESpaceWithLinearConstraints(aggdof_to_dof,aggdof_to_dofs,aggdof_to_coeffs,Vstd)
  U = TrialFESpace(V,u)

  a(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ
  l(v) = ∫( f*v )*dΩ + ∫( (g⋅n_Γ)*v )*dΓ

  op = AffineFEOperator(a,l,U,V)
  kop = cond(Array(get_matrix(op)))
  uh = solve(op)

  e = u - uh

  l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
  h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

  el2 = l2(e)
  eh1 = h1(e)

  ul2 = l2(uh)
  uh1 = h1(uh)

  num_free_dofs(U),el2,eh1,kop,ul2,uh1

end

function run_nodal(od::Int,κ::Real)

  function stretching(x::Fields.Point)
    m = zeros(length(x))
    m[1] = x[1] + hs(x[1])*κ
    Fields.Point(m)
  end

  n = 2; partition = (n); domain = (0,2)
  model = CartesianDiscreteModel(domain,partition,map=stretching)

  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)
  n_Γ = get_normal_vector(Γ)

  degree = 2*od
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  reffe = ReferenceFE(lagrangian,Float64,od)
  Vstd = TestFESpace(model,reffe,dirichlet_tags=[1])

  aggdof_to_dof = vcat(2,collect(2+od:2*od))

  aggdof_to_dofs_ptrs = [1+i*(od+1) for i in 0:od]
  aggdof_to_dofs_data = repeat(vcat(-1,1,collect(3:od+1)),od)
  aggdof_to_dofs = Table(aggdof_to_dofs_data,aggdof_to_dofs_ptrs)

  sh1 = get_cell_shapefuns(Vstd).cell_basis[1]
  df2 = get_cell_dof_basis(Vstd).cell_dof[2]
  ld2 = ( ( 1.0 + κ ) .* df2.nodes ) .+ 1.0
  ld2 = LagrangianDofBasis(df2,ld2)
  aggdof_to_coeffs_data = transpose(evaluate(ld2,sh1))[od+2:end]
  aggdof_to_coeffs = Table(aggdof_to_coeffs_data,aggdof_to_dofs_ptrs)

  V = FESpaceWithLinearConstraints(aggdof_to_dof,aggdof_to_dofs,aggdof_to_coeffs,Vstd)
  U = TrialFESpace(V,u)

  a(u,v) = ∫( ∇(u)⋅∇(v) )*dΩ
  l(v) = ∫( f*v )*dΩ + ∫( (g⋅n_Γ)*v )*dΓ

  op = AffineFEOperator(a,l,U,V)
  kop = cond(Array(get_matrix(op)))
  uh = solve(op)

  e = u - uh

  l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
  h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

  el2 = l2(e)
  eh1 = h1(e)

  num_free_dofs(U),el2,eh1,kop

end

function conv_test(np::Int,κ::Real)

  ps = Float64[]

  el2Ms = Float64[]
  eh1Ms = Float64[]
  kMs = Float64[]

  el2Ns = Float64[]
  eh1Ns = Float64[]
  kNs = Float64[]

  ul2 = 0.0
  uh1 = 0.0

  for p in 1:np

    N, el2, eh1, k, ul2, uh1 = run_modal(p,κ)

    push!(ps,N)

    push!(el2Ms,el2)
    push!(eh1Ms,eh1)
    push!(kMs,k)

    N, el2, eh1, k = run_nodal(p,κ)

    push!(el2Ns,el2)
    push!(eh1Ns,eh1)
    push!(kNs,k)

  end

  el2Ms = el2Ms ./ ul2
  eh1Ms = eh1Ms ./ uh1

  el2Ns = el2Ns ./ ul2
  eh1Ns = eh1Ns ./ uh1

  (ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs)

end

axE = layout[1,1] = LAxis(scene)
axC = layout[1,2] = LAxis(scene)

cols = [:red, :blue, :green, :orange, :purple, :brown]
lins = [nothing, :dash, :dot, :dashdot, :dashdotdot, nothing]

κ = collect(-0.5:0.5:4.0)
s = LSlider(scene,range=1:length(κ),startvalue=1,
            color_inactive=RGBf0(1.0,1.0,1.0))

datal2M = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(el2Ms)
end

datah1M = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(eh1Ms)
end

datacnM = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(kMs)
end

line1 = lines!(axE,datal2M,color=:red,linewidth=2)
scat1 = scatter!(axE,datal2M,color=:red,marker=:circle,markersize=6.0)

line2 = lines!(axE,datah1M,color=:red,linewidth=2)
scat2 = scatter!(axE,datah1M,color=:red,marker=:xcross,markersize=6.0)

line3 = lines!(axC,datacnM,color=:red,linewidth=2)
scat3 = scatter!(axC,datacnM,color=:red,marker=:diamond,markersize=6.0)

datal2N = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(el2Ns)
end

datah1N = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(eh1Ns)
end

datacnN = lift(s.value) do iκ
  ps, el2Ms, eh1Ms, kMs, el2Ns, eh1Ns, kNs = conv_test(maxod,κ[iκ])
  log10.(ps), log10.(kNs)
end

line4 = lines!(axE,datal2N,color=:blue,linewidth=2)
scat4 = scatter!(axE,datal2N,color=:blue,marker=:circle,markersize=6.0)

line5 = lines!(axE,datah1N,color=:blue,linewidth=2)
scat5 = scatter!(axE,datah1N,color=:blue,marker=:xcross,markersize=6.0)

line6 = lines!(axC,datacnN,color=:blue,linewidth=2)
scat6 = scatter!(axC,datacnN,color=:blue,marker=:diamond,markersize=6.0)

limits!(axE,-0.1,1.1,-16,4)
axE.xticks = 0.0:0.25:1.0
axE.yticks = -15.0:3.0:3.0
axE.xlabel="log10(N)"
axE.ylabel="log10(Rel error)"

limits!(axC,-0.1,1.1,-0.5,20.5)
axC.xticks = 0.0:0.25:1.0
axC.yticks = 0.0:1.0:20.0
axC.xlabel="log10(N)"
axC.ylabel="log10(Condition number)"

axE.xticksize = 2.0; axE.yticksize = 2.0
axE.xticklabelsize = 12.0; axE.yticklabelsize = 12.0

axC.xticksize = 2.0; axC.yticksize = 2.0
axC.xticklabelsize = 12.0; axC.yticklabelsize = 12.0

mark1 = MarkerElement(color=:gray,marker=:circle,strokecolor=:black)
mark2 = MarkerElement(color=:gray,marker=:xcross,strokecolor=:black)
mark3 = MarkerElement(color=:gray,marker=:diamond,strokecolor=:black)
legmarkers = [ line1, line4, mark1, mark2, mark3 ]
legnames = [ "Modal", "Nodal", "L2(e)", "H1(e)", "κ(A)" ]
leg = LLegend( scene, legmarkers, legnames, orientation = :horizontal )

layout[2, :] =
  hbox!( LText(scene, lift(x -> "κ: $(κ[x])", s.value), width = 60), s, leg )

record(scene,"ModalC0AgFEM1D.mp4",1:length(κ); framerate = 1) do i
  s.value = i
end
# scene
