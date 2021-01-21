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

const maxod = 12
u(x) = -x[1]^maxod+maxod*x[1]
g(x) = ∇(u)(x)
f(x) = -Δ(u)(x)

hs(x::Real) = x > 1.0 ? 1.0 : 0.0

outer_padding = 30
scene, layout = layoutscene(outer_padding, resolution = (800, 800),
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

end

function run_old_modal(od::Int,κ::Real)

  function stretching(x::Fields.Point)
    m = zeros(length(x))
    m[1] = x[1] + hs(x[1])*κ
    Fields.Point(m)
  end

  n = 2; partition = (n); domain = (0,2)
  model = CartesianDiscreteModel(domain,partition,map=stretching)

  bboxes = [ Fields.Point{1,Float64}[(0.0),(1.0),(0.0),(1.0),(0.0),(1.0)],
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

end

κ = collect(-0.95:0.05:4.0)
ax = layout[1,1] = LAxis(scene)

for od in 1:maxod

  datakM = [ (iκ+1,log10(run_modal(od,iκ))) for iκ in κ]
  datakN = [ (iκ+1,log10(run_nodal(od,iκ))) for iκ in κ]

  lines!(ax,datakM,color=:red,linewidth=2)
  lines!(ax,datakN,color=:blue,linestyle=:dash,linewidth=2)

end

limits!(ax,-0.1,5.1,-1.0,24.0)
ax.xticks = 0.0:0.5:5.0
ax.yticks = 0.0:2.0:24.0
ax.xlabel="Max aggregation distance"
ax.ylabel="log10(Condition number)"
ax.title="CN vs agg. distance and order: modal (red) vs nodal (blue)"

save("ModalC0AgFEM1D.png",scene)
scene
