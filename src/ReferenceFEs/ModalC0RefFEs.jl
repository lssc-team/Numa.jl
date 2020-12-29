struct ModalC0 <: ReferenceFEName end

const modalC0 = ModalC0()

"""
  ModalC0RefFE(::Type{T},p::Polytope{D},orders) where {T,D}

Returns an instance of `GenericRefFE{ModalC0}` representing a ReferenceFE with
Modal C0-continuous shape functions (multivariate scalar-valued, vector-valued,
or tensor-valued, iso- or aniso-tropic).

For more details about the shape functions, see Section 1.1.5 in

Ern, A., & Guermond, J. L. (2013). Theory and practice of finite elements
(Vol. 159). Springer Science & Business Media.

and references therein.

The constructor is only implemented for for n-cubes and the minimum order in
any dimension must be greater than one. The DoFs are numbered by n-faces in the
same way as with CLagrangianRefFEs.
"""
function ModalC0RefFE(
  ::Type{T},
  p::Polytope{D},
  orders;
  type::Symbol=:agfem,
  ξ₀::Point{D,V}=Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}())),
  ξ₁::Point{D,V}=Point{D,eltype(T)}(tfill(one(eltype(T)),Val{D}())) ) where {T,D,V}

  @notimplementedif ! is_n_cube(p)
  @notimplementedif minimum(orders) < one(eltype(orders))

  if ( type == :modified )
    shapefuns = ModifiedModalC0Basis{D}(T,orders,ξ₀=ξ₀,ξ₁=ξ₁)
  else
    shapefuns = AgFEMModalC0Basis{D}(T,orders)
  end

  nodes, face_own_nodes = compute_nodes(p,shapefuns.orders)
  predofs = LagrangianDofBasis(T,nodes)
  ndofs = length(predofs.dof_to_node)
  change = inv(evaluate(predofs,shapefuns))
  dofs = linear_combination(change,predofs)

  nnodes = length(predofs.nodes)
  reffaces = compute_lagrangian_reffaces(T,p,shapefuns.orders)
  _reffaces = vcat(reffaces...)
  face_nodes = _generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = _generate_face_own_dofs(face_own_nodes,predofs.node_and_comp_to_dof)
  face_dofs = _generate_face_dofs(ndofs,face_own_dofs,p,_reffaces)

  lag_reffe = ReferenceFE(p,lagrangian,T,orders)

  GenericRefFE{ModalC0}(
    ndofs,
    p,
    shapefuns,
    dofs,
    GradConformity(),
    lag_reffe,
    face_dofs,
    shapefuns)
end

function ReferenceFE(
  polytope::Polytope,
  ::ModalC0,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}};
  kwargs...) where T

  ModalC0RefFE(T,polytope,orders;kwargs...)
end

function Conformity(reffe::GenericRefFE{ModalC0},sym::Symbol)
  h1 = (:H1,:C0,:Hgrad)
  if sym == :L2
    L2Conformity()
  elseif sym in h1
    H1Conformity()
  else
    @unreachable """\n
    It is not possible to use conformity = $sym on a ModalC0RefFE with H1 conformity.
    Possible values of conformity for this reference fe are $((:L2, h1...)).
    """
  end
end

function get_face_own_dofs(
  reffe::GenericRefFE{ModalC0},conf::GradConformity)
  lagrangian_reffe = reffe.metadata
  get_face_own_dofs(lagrangian_reffe,conf)
end

function get_face_own_dofs_permutations(
  reffe::GenericRefFE{ModalC0},conf::GradConformity)
  lagrangian_reffe = reffe.metadata
  get_face_own_dofs_permutations(lagrangian_reffe,conf)
end
