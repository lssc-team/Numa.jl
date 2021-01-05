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
  orders,
  a::Vector{Point{D,V}},
  b::Vector{Point{D,V}} ) where {T,D,V}

  @notimplementedif ! is_n_cube(p)
  @notimplementedif minimum(orders) < one(eltype(orders))

  shapefuns = AgFEMModalC0Basis{D}(T,orders,a,b)

  ndofs, predofs, lag_reffe, face_dofs = compute_lag_reffe_data(T,p,orders)

  GenericRefFE{ModalC0}(
    ndofs,
    p,
    predofs,
    GradConformity(),
    lag_reffe,
    face_dofs,
    shapefuns)
end

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

  ndofs, predofs, lag_reffe, face_dofs = compute_lag_reffe_data(T,p,orders)

  GenericRefFE{ModalC0}(
    ndofs,
    p,
    predofs,
    GradConformity(),
    lag_reffe,
    face_dofs,
    shapefuns)
end

function compute_lag_reffe_data(::Type{T},
                                p::Polytope{D},
                                order::Int) where {T,D}

  orders = tfill(order,Val{D}())
  compute_lag_reffe_data(T,p,orders)
end

function compute_lag_reffe_data(::Type{T},
                                p::Polytope{D},
                                orders::NTuple{D,Int}) where {T,D}

  nodes, face_own_nodes = compute_nodes(p,orders)
  predofs = LagrangianDofBasis(T,nodes)
  ndofs = length(predofs.dof_to_node)

  nnodes = length(predofs.nodes)
  reffaces = compute_lagrangian_reffaces(T,p,orders)
  _reffaces = vcat(reffaces...)
  face_nodes = _generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = _generate_face_own_dofs(face_own_nodes,predofs.node_and_comp_to_dof)
  face_dofs = _generate_face_dofs(ndofs,face_own_dofs,p,_reffaces)

  lag_reffe = ReferenceFE(p,lagrangian,T,orders)
  ndofs, predofs, lag_reffe, face_dofs
end

function ReferenceFE(
  polytope::Polytope{D},
  ::ModalC0,
  ::Type{T},
  orders::Union{Integer,NTuple{D,Int}};
  kwargs...) where {T,D}

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

function compute_shapefun_bboxes!(
  a::Vector{Point{D,V}},
  b::Vector{Point{D,V}},
  bboxes::Vector{Point{D,V}},
  face_own_dofs) where {D,V}
  for i in 1:length(face_own_dofs)
    a[face_own_dofs[i]] .= bboxes[2*i-1]
    b[face_own_dofs[i]] .= bboxes[2*i]
  end
end

function compute_cell_to_modalC0_reffe(
  p::Polytope{D},
  ::Type{T},
  orders::Union{Integer,NTuple{D,Int}},
  bboxes) where {T,D} # type-stability?

  @notimplementedif ! is_n_cube(p)
  @notimplementedif minimum(orders) < one(eltype(orders))

  ndofs, predofs, lag_reffe, face_dofs = compute_lag_reffe_data(T,p,orders)
  face_own_dofs = get_face_own_dofs(lag_reffe,GradConformity())

  sh(bbs) = begin
    a = fill(Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}())),ndofs)
    b = fill(Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}())),ndofs)
    compute_shapefun_bboxes!(a,b,bbs,face_own_dofs)
    AgFEMModalC0Basis{D}(T,orders,a,b)
  end

  reffe(sh) = GenericRefFE{ModalC0}(ndofs,
                                    p,
                                    predofs,
                                    GradConformity(),
                                    lag_reffe,
                                    face_dofs,
                                    sh)

  reffes = [ reffe(sh(bbs)) for bbs in bboxes ]
  CompressedArray(reffes,1:length(reffes))
end
