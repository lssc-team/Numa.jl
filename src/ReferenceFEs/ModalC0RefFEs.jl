function ModalC0RefFE(::Type{T},p::Polytope{D},orders) where {T,D}

  shapefuns = ModalC0Basis{D}(T,orders) # Have ModalC0Basis(T,p,orders)?
  nodes, face_own_nodes = compute_nodes(p,shapefuns.orders)
  predofs = LagrangianDofBasis(T,nodes)
  ndofs = length(predofs.dof_to_node)
  change = inv(evaluate(predofs,shapefuns))
  dofs = linear_combination(change,predofs)

  # 1. Encapsulate and use by both Lag and Modal
  nnodes = length(predofs.nodes)
  reffaces = compute_lagrangian_reffaces(T,p,shapefuns.orders)
  _reffaces = vcat(reffaces...)
  face_nodes = _generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = _generate_face_own_dofs(face_own_nodes,predofs.node_and_comp_to_dof)
  face_dofs = _generate_face_dofs(ndofs,face_own_dofs,p,_reffaces)

  lag_reffe = ReferenceFE(p,:Lagrangian,T,orders)

  # 1. Review parametric symbol
  # 2. GradConformity because only implemented for order > 0
  GenericRefFE{:ModalC0}(
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
  ::Val{:ModalC0},
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}}) where T

  ModalC0RefFE(T,polytope,orders)
end

function get_face_own_dofs(
  reffe::GenericRefFE{:ModalC0},conf::GradConformity)
  lagrangian_reffe = reffe.metadata
  get_face_own_dofs(lagrangian_reffe,conf)
end

function get_face_own_dofs_permutations(
  reffe::GenericRefFE{:ModalC0},conf::GradConformity)
  lagrangian_reffe = reffe.metadata
  get_face_own_dofs_permutations(lagrangian_reffe,conf)
end
