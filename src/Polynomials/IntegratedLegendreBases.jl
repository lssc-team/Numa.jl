struct IntegratedLegendre <: Field end

struct IntegratedLegendreBasis{D,T} <: AbstractVector{IntegratedLegendre}
  orders::NTuple{D,Int}
  terms::Vector{CartesianIndex{D}}
  function IntegratedLegendreBasis{D}(
    ::Type{T}, orders::NTuple{D,Int}, terms::Vector{CartesianIndex{D}}) where {D,T}
    new{D,T}(orders,terms)
  end
end

@inline Base.size(a::IntegratedLegendreBasis{D,T}) where {D,T} = (length(a.terms)*num_components(T),)
@inline Base.getindex(a::IntegratedLegendreBasis,i::Integer) = IntegratedLegendre()
@inline Base.IndexStyle(::IntegratedLegendreBasis) = IndexLinear()

function IntegratedLegendreBasis{D}(
  ::Type{T}, orders::NTuple{D,Int}; filter::Function=_q_filter_il, sort!::Function=_sort_by_nfaces!) where {D,T}

  terms = _define_terms_il(filter, sort!, orders)
  IntegratedLegendreBasis{D}(T,orders,terms)
end

function IntegratedLegendreBasis{D}(
  ::Type{T}, order::Int; filter::Function=_q_filter_il, sort!::Function=_sort_by_nfaces!) where {D,T}

  orders = tfill(order,Val{D}())
  IntegratedLegendreBasis{D}(T,orders;filter=filter,sort! = sort!)
end

# API

function get_exponents(b::IntegratedLegendreBasis)
  indexbase = 1
  [Tuple(t) .- indexbase for t in b.terms]
end

function get_order(b::IntegratedLegendreBasis)
  maximum(b.orders)
end

function get_orders(b::IntegratedLegendreBasis)
  b.orders
end

return_type(::IntegratedLegendreBasis{D,T}) where {D,T} = T

# Field implementation

function return_cache(f::IntegratedLegendreBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c)
end

function evaluate!(cache,f::IntegratedLegendreBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  r, v, c = cache
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _evaluate_nd_il!(v,xi,f.orders,f.terms,c)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{1,IntegratedLegendreBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}

  f = fg.fa
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(V)
  xi = testitem(x)
  T = gradient_type(V,xi)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  g = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c, g)
end

function evaluate!(
  cache,
  fg::FieldGradientArray{1,IntegratedLegendreBasis{D,T}},
  x::AbstractVector{<:Point}) where {D,T}

  f = fg.fa
  r, v, c, g = cache
  np = length(x)
  ndof = length(f.terms) * num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  setsize!(g,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _gradient_nd_il!(v,xi,f.orders,f.terms,c,g,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{2,IntegratedLegendreBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}

  f = fg.fa
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(V)
  xi = testitem(x)
  T = gradient_type(gradient_type(V,xi),xi)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  g = CachedArray(zeros(eltype(T),(D,n)))
  h = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c, g, h)
end

function evaluate!(
  cache,
  fg::FieldGradientArray{2,IntegratedLegendreBasis{D,T}},
  x::AbstractVector{<:Point}) where {D,T}

  f = fg.fa
  r, v, c, g, h = cache
  np = length(x)
  ndof = length(f.terms) * num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  setsize!(g,(D,n))
  setsize!(h,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _hessian_nd_il!(v,xi,f.orders,f.terms,c,g,h,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

# Helpers

_q_filter_il(e,o) = true

_sort_by_tensor_prod!(terms,orders) = terms

function _sort_by_nfaces!(terms::Vector{CartesianIndex{D}},orders) where D

  # Generate indices of n-faces and order s.t.
  # (1) dimension-increasing (2) lexicographic
  bin_rang_nfaces = tfill(0:1,Val{D}())
  bin_ids_nfaces = collect(Iterators.product(bin_rang_nfaces...))
  sum_bin_ids_nfaces = [sum(bin_ids_nfaces[i]) for i in eachindex(bin_ids_nfaces)]
  bin_ids_nfaces = permute!(bin_ids_nfaces,sortperm(sum_bin_ids_nfaces))

  # Generate LIs of basis funs s.t. order by n-faces
  lids_b = LinearIndices(Tuple([orders[i]+1 for i=1:D]))

  eet = eltype(eltype(bin_ids_nfaces))
  f(x) = Tuple( x[i] == one(eet) ? (0:0) : (1:2) for i in 1:length(x) )
  g(x) = Tuple( x[i] == one(eet) ? (3:orders[i]+1) : (0:0) for i in 1:length(x) )
  rang_nfaces = map(f,bin_ids_nfaces)
  rang_own_dofs = map(g,bin_ids_nfaces)

  P = Int64[]
  for i = 1:length(bin_ids_nfaces)
    cis_nfaces = CartesianIndices(rang_nfaces[i])
    cis_own_dofs = CartesianIndices(rang_own_dofs[i])
    for ci in cis_nfaces
      ci = ci .+ cis_own_dofs
      P = vcat(P,reshape(lids_b[ci],length(ci)))
    end
  end

  permute!(terms,P)
end

function _define_terms_il(filter,sort!,orders)
  terms = _define_terms(filter,orders)
  sort!(terms,orders)
end

function _legendre(ξ,::Val{N}) where N
  ((2*N-1)*ξ*_legendre(ξ,Val{N-1}())-(N-1)*_legendre(ξ,Val{N-2}()))/N
end

_legendre(ξ,::Val{0}) = 1
_legendre(ξ,::Val{1}) = ξ
# TO-DO: Complete till 10

function _evaluate_1d_il!(v::AbstractMatrix{T},x,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = 1 - x[d]
  @inbounds v[d,2] = x[d]
  ξ = -1 + 2*x[d]
  for i in 3:n
    @inbounds v[d,i] = (_legendre(ξ,Val{i-1}())-_legendre(ξ,Val{i-3}()))/(2*sqrt(2*i-3))
  end
end

function _gradient_1d_il!(v::AbstractMatrix{T},x,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = -1
  @inbounds v[d,2] = 1
  ξ = -1 + 2*x[d]
  for i in 3:n
    @inbounds v[d,i] = sqrt(2*i-3)*_legendre(ξ,Val{i-2}())
  end
end

function _hessian_1d_il!(v::AbstractMatrix{T},x,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = 0
  @inbounds v[d,2] = 0
  ξ = -1 + 2*x[d]
  for i in 3:n
    @inbounds v[d,i] = sqrt(2*i-3)*2*((i-2)*_legendre(ξ,Val{i-3}())+ξ*v[d,i-1]/(2*sqrt(2*i-5)))
  end
end

function _evaluate_nd_il!(
  v::AbstractVector{V},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d_il!(c,x,orders[d],d)
  end

  o = one(T)
  k = 1

  for ci in terms

    s = o
    for d in 1:dim
      @inbounds s *= c[d,ci[d]]
    end

    k = _set_value!(v,s,k)

  end

end

function _gradient_nd_il!(
  v::AbstractVector{G},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d_il!(c,x,orders[d],d)
    _gradient_1d_il!(g,x,orders[d],d)
  end

  z = zero(Mutable(VectorValue{D,T}))
  o = one(T)
  k = 1

  for ci in terms

    s = z
    for i in eachindex(s)
      @inbounds s[i] = o
    end
    for q in 1:dim
      for d in 1:dim
        if d != q
          @inbounds s[q] *= c[d,ci[d]]
        else
          @inbounds s[q] *= g[d,ci[d]]
        end
      end
    end

    k = _set_gradient!(v,s,k,V)

  end

end

function _hessian_nd_il!(
  v::AbstractVector{G},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  h::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d_il!(c,x,orders[d],d)
    _gradient_1d_il!(g,x,orders[d],d)
    _hessian_1d_il!(h,x,orders[d],d)
  end

  z = zero(Mutable(TensorValue{D,D,T}))
  o = one(T)
  k = 1

  for ci in terms

    s = z
    for i in eachindex(s)
      @inbounds s[i] = o
    end
    for r in 1:dim
      for q in 1:dim
        for d in 1:dim
          if d != q && d != r
            @inbounds s[r,q] *= c[d,ci[d]]
          elseif d == q && d ==r
            @inbounds s[r,q] *= h[d,ci[d]]
          else
            @inbounds s[r,q] *= g[d,ci[d]]
          end
        end
      end
    end

    k = _set_gradient!(v,s,k,V)

  end

end
