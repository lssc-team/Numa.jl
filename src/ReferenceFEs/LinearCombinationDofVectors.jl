struct Mode <: Dof end

struct LinearCombinationDofVector{P,V,T} <: AbstractVector{Mode}
  lag_dof_basis::LagrangianDofBasis{P,V}
  change_of_basis::Matrix{T}
end

function LinearCombinationDofVector(::Type{T},p::Polytope,b::ModalC0Basis) where T
  lag_nodes, _ = compute_nodes(p,b.orders)
  lag_dof_basis = LagrangianDofBasis(T,lag_nodes)
  change_of_basis = inv(evaluate(lag_dof_basis,b))
  LinearCombinationDofVector(lag_dof_basis,change_of_basis)
end

@inline Base.size(a::LinearCombinationDofVector) = size(a.lag_dof_basis)
@inline Base.axes(a::LinearCombinationDofVector) = axes(a.lag_dof_basis)
@inline Base.getindex(a::LinearCombinationDofVector,i::Integer) = Mode()
@inline Base.IndexStyle(::LinearCombinationDofVector) = IndexLinear()

function return_cache(b::LinearCombinationDofVector,field)
  return_cache(b.lag_dof_basis,field)
end

@inline function evaluate!(cache,b::LinearCombinationDofVector,field)
  c, cf = cache
  vals = evaluate!(cf,field,b.lag_dof_basis.nodes)
  vals = evaluate!(c,*,b.change_of_basis,vals)
  ndofs = length(b.lag_dof_basis.dof_to_node)
  T = eltype(vals)
  ncomps = num_components(T)
  @check ncomps == num_components(eltype(b.lag_dof_basis.node_and_comp_to_dof)) """\n
  Unable to evaluate LagrangianDofBasis. The number of components of the
  given Field does not match with the LagrangianDofBasis.

  If you are trying to interpolate a function on a FESpace make sure that
  both objects have the same value type.

  For instance, trying to interpolate a vector-valued funciton on a scalar-valued FE space
  would raise this error.
  """
  _evaluate_lagr_dof!(c,vals,b.lag_dof_basis.node_and_comp_to_dof,ndofs,ncomps)
end
