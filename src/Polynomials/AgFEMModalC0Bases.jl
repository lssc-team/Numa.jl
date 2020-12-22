struct AgFEMModalC0 <: Field end

struct AgFEMModalC0Basis{D,T,V} <: AbstractVector{AgFEMModalC0}
  orders::NTuple{D,Int}
  terms::Vector{CartesianIndex{D}}
  a::Vector{Point{D,V}}
  b::Vector{Point{D,V}}
  function AgFEMModalC0Basis{D}(
    ::Type{T},
    orders::NTuple{D,Int},
    terms::Vector{CartesianIndex{D}},
    a::Vector{Point{D,V}},
    b::Vector{Point{D,V}}) where {D,T,V}
    new{D,T,V}(orders,terms,a,b)
  end
end

@inline Base.size(a::AgFEMModalC0Basis{D,T,V}) where {D,T,V} = (length(a.terms)*num_components(T),)
@inline Base.getindex(a::AgFEMModalC0Basis,i::Integer) = AgFEMModalC0()
@inline Base.IndexStyle(::AgFEMModalC0Basis) = IndexLinear()

function AgFEMModalC0Basis{D}(
  ::Type{T},
  orders::NTuple{D,Int},
  a::Vector{Point{D,V}},
  b::Vector{Point{D,V}};
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T,V}

  terms = _define_terms_mc0(filter, sort!, orders)
  AgFEMModalC0Basis{D}(T,orders,terms,a,b)
end

function AgFEMModalC0Basis{D}(
  ::Type{T},
  orders::NTuple{D,Int},
  sa::Point{D,V},
  sb::Point{D,V};
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T,V}

  terms = _define_terms_mc0(filter, sort!, orders)
  a = fill(sa,length(terms))
  b = fill(sb,length(terms))
  AgFEMModalC0Basis{D}(T,orders,terms,a,b)
end

function AgFEMModalC0Basis{D}(
  ::Type{T},
  orders::NTuple{D,Int};
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T}

  sa = Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}()))
  sb = Point{D,eltype(T)}(tfill(one(eltype(T)),Val{D}()))
  AgFEMModalC0Basis{D}(T,orders,sa,sb,filter=filter,sort! = sort!)
end

function AgFEMModalC0Basis{D}(
  ::Type{T},
  order::Int,
  a::Vector{Point{D,V}},
  b::Vector{Point{D,V}};
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T,V}

  orders = tfill(order,Val{D}())
  AgFEMModalC0Basis{D}(T,orders,a,b,filter=filter,sort! = sort!)
end

function AgFEMModalC0Basis{D}(
  ::Type{T},
  order::Int;
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T}

  orders = tfill(order,Val{D}())
  AgFEMModalC0Basis{D}(T,orders,filter=filter,sort! = sort!)
end

# API

"""
    get_order(b::AgFEMModalC0Basis)
"""
function get_order(b::AgFEMModalC0Basis)
  maximum(b.orders)
end

"""
    get_orders(b::AgFEMModalC0Basis)
"""
function get_orders(b::AgFEMModalC0Basis)
  b.orders
end

return_type(::AgFEMModalC0Basis{D,T,V}) where {D,T,V} = T

# Field implementation

function return_cache(f::AgFEMModalC0Basis{D,T,V},x::AbstractVector{<:Point}) where {D,T,V}
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c)
end

function evaluate!(cache,f::AgFEMModalC0Basis{D,T,V},x::AbstractVector{<:Point}) where {D,T,V}
  r, v, c = cache
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _evaluate_nd_amc0!(v,xi,f.a,f.b,f.orders,f.terms,c)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{1,AgFEMModalC0Basis{D,V,W}},
  x::AbstractVector{<:Point}) where {D,V,W}

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
  fg::FieldGradientArray{1,AgFEMModalC0Basis{D,T,V}},
  x::AbstractVector{<:Point}) where {D,T,V}

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
    _gradient_nd_amc0!(v,xi,f.a,f.b,f.orders,f.terms,c,g,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{2,AgFEMModalC0Basis{D,V,W}},
  x::AbstractVector{<:Point}) where {D,V,W}

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
  fg::FieldGradientArray{2,AgFEMModalC0Basis{D,T,V}},
  x::AbstractVector{<:Point}) where {D,T,V}

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
    _hessian_nd_amc0!(v,xi,f.a,f.b,f.orders,f.terms,c,g,h,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

# Helpers

function _evaluate_1d_amc0!(v::AbstractMatrix{T},x,a,b,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = z - x[d]
  @inbounds v[d,2] = x[d]
  if n > 2
    ξ = ( 2*x[d] - ( a[d] + b[d] ) ) / ( b[d] - a[d] )
    for i in 3:n
      @inbounds v[d,i] = -sqrt(2*i-3)*v[d,1]*v[d,2]*jacobi(ξ,i-3,1,1)/(i-2)
    end
  end
end

function _gradient_1d_amc0!(v::AbstractMatrix{T},x,a,b,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = -z
  @inbounds v[d,2] = z
  if n > 2
    ξ = ( 2*x[d] - ( a[d] + b[d] ) ) / ( b[d] - a[d] )
    v1 = z - x[d]
    v2 = x[d]
    for i in 3:n
      j, dj = jacobi_and_derivative(ξ,i-3,1,1)
      @inbounds v[d,i] = -sqrt(2*i-3)*(v[d,1]*v2*j+v1*v[d,2]*j+v1*v2*(2/(b[d]-a[d]))*dj)/(i-2)
    end
  end
end

function _hessian_1d_amc0!(v::AbstractMatrix{T},x,a,b,order,d) where T
  @assert order > 0
  n = order + 1
  y = zero(T)
  z = one(T)
  @inbounds v[d,1] = y
  @inbounds v[d,2] = y
  if n > 2
    ξ = ( 2*x[d] - ( a[d] + b[d] ) ) / ( b[d] - a[d] )
    v1 = z - x[d]
    v2 = x[d]
    dv1 = -z
    dv2 = z
    for i in 3:n
      j, dj = jacobi_and_derivative(ξ,i-3,1,1)
      _, d2j = jacobi_and_derivative(ξ,i-4,2,2)
      @inbounds v[d,i] = -sqrt(2*i-3)*(2*dv1*dv2*j+2*(dv1*v2+v1*dv2)*(2/(b[d]-a[d]))*dj+v1*v2*d2j*2*i*((b[d]-a[d])^2))/(i-2)
    end
  end
end

function _evaluate_nd_amc0!(
  v::AbstractVector{V},
  x,
  a::Vector{Point{D,T}},
  b::Vector{Point{D,T}},
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  o = one(T)
  k = 1
  l = length(terms)

  for (i,ci) in enumerate(terms)

    for d in 1:dim
      _evaluate_1d_amc0!(c,x,a[i],b[i],orders[d],d)
    end

    s = o
    for d in 1:dim
      @inbounds s *= c[d,ci[d]]
    end

    k = _set_value_mc0!(v,s,k,l)

  end

end

function _gradient_nd_amc0!(
  v::AbstractVector{G},
  x,
  a::Vector{Point{D,T}},
  b::Vector{Point{D,T}},
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  z = zero(Mutable(VectorValue{D,T}))
  o = one(T)
  k = 1
  l = length(terms)

  for (i,ci) in enumerate(terms)

    for d in 1:dim
      _evaluate_1d_amc0!(c,x,a[i],b[i],orders[d],d)
      _gradient_1d_amc0!(g,x,a[i],b[i],orders[d],d)
    end

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

function _hessian_nd_amc0!(
  v::AbstractVector{G},
  x,
  a::Vector{Point{D,T}},
  b::Vector{Point{D,T}},
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  h::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  z = zero(Mutable(TensorValue{D,D,T}))
  o = one(T)
  k = 1
  l = length(terms)

  for (i,ci) in enumerate(terms)

    for d in 1:dim
      _evaluate_1d_amc0!(c,x,a[i],b[i],orders[d],d)
      _gradient_1d_amc0!(g,x,a[i],b[i],orders[d],d)
      _hessian_1d_amc0!(h,x,a[i],b[i],orders[d],d)
    end

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
