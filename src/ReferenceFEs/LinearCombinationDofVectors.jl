function linear_combination(a::AbstractMatrix{<:Number},
                            b::AbstractVector{<:Dof})
  LinearCombinationDofVector(a,b)
end

struct LinearCombinationDofVector{T} <: AbstractVector{Dof}
  change_of_basis::Matrix{T}
  dof_basis::AbstractVector{<:Dof}
end

@inline Base.size(a::LinearCombinationDofVector) = size(a.dof_basis)
@inline Base.axes(a::LinearCombinationDofVector) = axes(a.dof_basis)
@inline Base.getindex(a::LinearCombinationDofVector,i::Integer) = getindex(a.dof_basis,i)
@inline Base.IndexStyle(::LinearCombinationDofVector) = IndexLinear()

function return_cache(b::LinearCombinationDofVector,field)
  # vals_to_lincom_vals = linear_combination(b.change_of_basis,field)
  # return_cache(b.dof_basis,vals_to_lincom_vals)
  c, cf = return_cache(b.dof_basis,field)
  c, cf, return_cache(*,b.change_of_basis,c)
end

@inline function evaluate!(cache,b::LinearCombinationDofVector,field)
  # vals_to_lincom_vals = linear_combination(b.change_of_basis,field)
  # evaluate!(cache,b.dof_basis,vals_to_lincom_vals)
  c, cf, cc = cache
  vals = evaluate!(cache,b.dof_basis,field)
  evaluate!(cc,*,b.change_of_basis,vals)
end

function test_lincom_dofvecs(::Type{T},p::Polytope{D},orders::NTuple{D,Int}) where {D,T}
  mb = ModalC0Basis{D}(T,orders)
  nodes, _ = compute_nodes(p,mb.orders)
  predofs = LagrangianDofBasis(T,nodes)
  change = inv(evaluate(predofs,mb))
  lincom_dofvals = linear_combination(change,predofs)
  id = Matrix{eltype(T)}(I,size(mb)[1],size(mb)[1])
  @test id ≈ evaluate(lincom_dofvals,mb)
end

function test_lincom_dofvecs(::Type{T},p::Polytope{D},order::Int) where {D,T}
  orders = tfill(order,Val{D}())
  test_lincom_dofvecs(T,p,orders)
end
