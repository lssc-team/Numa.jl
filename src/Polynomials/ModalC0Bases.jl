struct ModalC0 <: Field end

"""
    struct ModalC0Basis{D,T} <: AbstractVector{ModalC0}

Type representing a basis of multivariate scalar-valued, vector-valued, or
tensor-valued, iso- or aniso-tropic Modal C0-continuous polynomials (aka
integrated Legendre). The fields of this `struct` are not public.

See Section 1.1.5 in

Ern, A., & Guermond, J. L. (2013). Theory and practice of finite elements
(Vol. 159). Springer Science & Business Media.

and references therein.

This type fully implements the [`Field`](@ref) interface, with up to second
order derivatives.
"""
struct ModalC0Basis{D,T} <: AbstractVector{ModalC0}
  orders::NTuple{D,Int}
  terms::Vector{CartesianIndex{D}}
  function ModalC0Basis{D}(
    ::Type{T}, orders::NTuple{D,Int}, terms::Vector{CartesianIndex{D}}) where {D,T}
    new{D,T}(orders,terms)
  end
end

@inline Base.size(a::ModalC0Basis{D,T}) where {D,T} = (length(a.terms)*num_components(T),)
@inline Base.getindex(a::ModalC0Basis,i::Integer) = ModalC0()
@inline Base.IndexStyle(::ModalC0Basis) = IndexLinear()

"""
    ModalC0Basis{D}(::Type{T}, orders::Tuple [, filter::Function, sort!::Function]) where {D,T}

This version of the constructor allows to pass a tuple `orders` containing the
polynomial order to be used in each of the `D` dimensions in order to construct
and anisotropic tensor-product space.
"""
function ModalC0Basis{D}(
  ::Type{T}, orders::NTuple{D,Int}; filter::Function=_q_filter_mc0, sort!::Function=_sort_by_nfaces!) where {D,T}

  terms = _define_terms_mc0(filter, sort!, orders)
  ModalC0Basis{D}(T,orders,terms)
end

"""
    ModalC0Basis{D}(::Type{T}, order::Int [, filter::Function, sort!::Function]) where {D,T}

Returns an instance of `ModalC0Basis` representing a multivariate Modal C0-continuous
polynomial basis in `D` dimensions, of polynomial degree `order`, whose value is represented
by the type `T`. The type `T` is typically `<:Number`, e.g., `Float64` for scalar-valued
functions and `VectorValue{D,Float64}` for vector-valued ones.

# Filter function

The `filter` function is used to select which terms of the tensor product space
of order `order` in `D` dimensions are to be used. If the filter is not provided,
the full tensor-product space is used by default leading to a multivariate polynomial
space of type Q.

The signature of the filter function is

    (e,order) -> Bool

where `e` is a tuple of `D` integers containing the exponents of a multivariate
monomial. The following filters are used to select well known polynomial spaces

- Q space: `(e,order) -> true`
- P space: `(e,order) -> sum(e) <= order`
- "Serendipity" space: `(e,order) -> sum( [ i for i in e if i>1 ] ) <= order`

# Sort! function

The `sort!` function is used to sort the terms of the basis functions according
to the user needs. The following sort functions have been implemented:

- _sort_by_nfaces! (default): Orders the terms by the n-faces of the ReferenceFE,
                              such that the local DoF numbering coincides with
                              the one of Lagrangian FEs
- _sort_by_tensor_prod!: Orders the terms by the cartesian indices of the
                         tensor product of the 1D basis functions
"""
function ModalC0Basis{D}(
  ::Type{T}, order::Int; filter::Function=_q_filter_mc0, sort!::Function=_sort_by_nfaces!) where {D,T}

  orders = tfill(order,Val{D}())
  ModalC0Basis{D}(T,orders;filter=filter,sort! = sort!)
end

# API

"""
    get_order(b::ModalC0Basis)
"""
function get_order(b::ModalC0Basis)
  maximum(b.orders)
end

"""
    get_orders(b::ModalC0Basis)
"""
function get_orders(b::ModalC0Basis)
  b.orders
end

return_type(::ModalC0Basis{D,T}) where {D,T} = T

# Field implementation

function return_cache(f::ModalC0Basis{D,T},x::AbstractVector{<:Point}) where {D,T}
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c)
end

function evaluate!(cache,f::ModalC0Basis{D,T},x::AbstractVector{<:Point}) where {D,T}
  r, v, c = cache
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _evaluate_nd_mc0!(v,xi,f.orders,f.terms,c)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{1,ModalC0Basis{D,V}},
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
  fg::FieldGradientArray{1,ModalC0Basis{D,T}},
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
    _gradient_nd_mc0!(v,xi,f.orders,f.terms,c,g,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{2,ModalC0Basis{D,V}},
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
  fg::FieldGradientArray{2,ModalC0Basis{D,T}},
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
    _hessian_nd_mc0!(v,xi,f.orders,f.terms,c,g,h,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

# Helpers

_q_filter_mc0(e,o) = true

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

function _define_terms_mc0(filter,sort!,orders)
  terms = _define_terms(filter,orders)
  sort!(terms,orders)
end

function _legendre(ξ,::Val{N}) where N
  ((2*N-1)*ξ*_legendre(ξ,Val{N-1}())-(N-1)*_legendre(ξ,Val{N-2}()))/N
end

_legendre(ξ,::Val{0}) = 1
_legendre(ξ,::Val{1}) = ξ
_legendre(ξ,::Val{2}) = 0.5*(3*ξ^2-1)
_legendre(ξ,::Val{3}) = 0.5*(5*ξ^3-3*ξ)
_legendre(ξ,::Val{4}) = 0.125*(35*ξ^4-30*ξ^2+3)
_legendre(ξ,::Val{5}) = 0.125*(63*ξ^5-70*ξ^3+15*ξ)
_legendre(ξ,::Val{6}) = 0.0625*(231*ξ^6-315*ξ^4+105*ξ^2-5)
_legendre(ξ,::Val{7}) = 0.0625*(429*ξ^7-693*ξ^5+315*ξ^3-35*ξ)
_legendre(ξ,::Val{8}) = 0.0078125*(6435*ξ^8-12012*ξ^6+6930*ξ^4-1260*ξ^2+35)
_legendre(ξ,::Val{9}) = 0.0078125*(12155*ξ^9-25740*ξ^7+18018*ξ^5-4620*ξ^3+315*ξ)
_legendre(ξ,::Val{10}) = 0.00390625*(46189*ξ^10-109395*ξ^8+90090*ξ^6-30030*ξ^4+3465*ξ^2-63)

function _evaluate_1d_mc0!(v::AbstractMatrix{T},x,order,d) where T
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

function _gradient_1d_mc0!(v::AbstractMatrix{T},x,order,d) where T
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

function _hessian_1d_mc0!(v::AbstractMatrix{T},x,order,d) where T
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

function _evaluate_nd_mc0!(
  v::AbstractVector{V},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d_mc0!(c,x,orders[d],d)
  end

  o = one(T)
  k = 1
  l = length(terms)

  for ci in terms

    s = o
    for d in 1:dim
      @inbounds s *= c[d,ci[d]]
    end

    k = _set_value_mc0!(v,s,k,l)

  end

end

@inline function _set_value_mc0!(v::AbstractVector{V},s::T,k,l) where {V,T}
  m = zero(Mutable(V))
  z = zero(T)
  js = eachindex(m)
  for j in js
    for i in js
      @inbounds m[i] = z
    end
    @inbounds m[j] = s
    i = k+l*(j-1)
    @inbounds v[i] = m
  end
  k+1
end

@inline function _set_value_mc0!(v::AbstractVector{<:Real},s,k,l)
  @inbounds v[k] = s
  k+1
end

function _gradient_nd_mc0!(
  v::AbstractVector{G},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d_mc0!(c,x,orders[d],d)
    _gradient_1d_mc0!(g,x,orders[d],d)
  end

  z = zero(Mutable(VectorValue{D,T}))
  o = one(T)
  k = 1
  l = length(terms)

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

    k = _set_gradient_mc0!(v,s,k,l,V)

  end

end

@inline function _set_gradient_mc0!(
  v::AbstractVector{G},s,k,l,::Type{<:Real}) where G

  @inbounds v[k] = s
  k+1
end

@inline function _set_gradient_mc0!(
  v::AbstractVector{G},s,k,l,::Type{V}) where {V,G}

  T = eltype(s)
  m = zero(Mutable(G))
  w = zero(V)
  z = zero(T)
  for (ij,j) in enumerate(CartesianIndices(w))
    for i in CartesianIndices(m)
      @inbounds m[i] = z
    end
    for i in CartesianIndices(s)
      @inbounds m[i,j] = s[i]
    end
    i = k+l*(ij-1)
    @inbounds v[i] = m
  end
  k+1
end

function _hessian_nd_mc0!(
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
    _evaluate_1d_mc0!(c,x,orders[d],d)
    _gradient_1d_mc0!(g,x,orders[d],d)
    _hessian_1d_mc0!(h,x,orders[d],d)
  end

  z = zero(Mutable(TensorValue{D,D,T}))
  o = one(T)
  k = 1
  l = length(terms)

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

    k = _set_gradient_mc0!(v,s,k,l,V)

  end

end
