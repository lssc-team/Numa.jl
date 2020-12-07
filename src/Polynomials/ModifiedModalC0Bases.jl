struct ModifiedModalC0 <: Field end

struct ModifiedModalC0Basis{D,T,V} <: AbstractVector{ModifiedModalC0}
  orders::NTuple{D,Int}
  terms::Vector{CartesianIndex{D}}
  ξ₀::Point{D,V}
  ξ₁::Point{D,V}
  function ModifiedModalC0Basis{D}(
    ::Type{T},
    orders::NTuple{D,Int},
    terms::Vector{CartesianIndex{D}},
    ξ₀::Point{D,V},
    ξ₁::Point{D,V}) where {D,T,V}
    new{D,T,V}(orders,terms,ξ₀,ξ₁)
  end
end

@inline Base.size(a::ModifiedModalC0Basis{D,T,V}) where {D,T,V} = (length(a.terms)*num_components(T),)
@inline Base.getindex(a::ModifiedModalC0Basis,i::Integer) = ModifiedModalC0()
@inline Base.IndexStyle(::ModifiedModalC0Basis) = IndexLinear()

function ModifiedModalC0Basis{D}(
  ::Type{T},
  orders::NTuple{D,Int};
  ξ₀::Point{D,V}=Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}())),
  ξ₁::Point{D,V}=Point{D,eltype(T)}(tfill(one(eltype(T)),Val{D}())),
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T,V}

  terms = _define_terms_mc0(filter, sort!, orders)
  ModifiedModalC0Basis{D}(T,orders,terms,ξ₀,ξ₁)
end

function ModifiedModalC0Basis{D}(
  ::Type{T},
  order::Int;
  ξ₀::Point{D,V}=Point{D,eltype(T)}(tfill(zero(eltype(T)),Val{D}())),
  ξ₁::Point{D,V}=Point{D,eltype(T)}(tfill(one(eltype(T)),Val{D}())),
  filter::Function=_q_filter_mc0,
  sort!::Function=_sort_by_nfaces!) where {D,T,V}

  orders = tfill(order,Val{D}())
  ModifiedModalC0Basis{D}(T,orders,ξ₀=ξ₀,ξ₁=ξ₁,filter=filter,sort! = sort!)
end

# API

"""
    get_order(b::ModifiedModalC0Basis)
"""
function get_order(b::ModifiedModalC0Basis)
  maximum(b.orders)
end

"""
    get_orders(b::ModifiedModalC0Basis)
"""
function get_orders(b::ModifiedModalC0Basis)
  b.orders
end

return_type(::ModifiedModalC0Basis{D,T,V}) where {D,T,V} = T

# Field implementation

function return_cache(f::ModifiedModalC0Basis{D,T,V},x::AbstractVector{<:Point}) where {D,T,V}
  @assert D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c)
end

function evaluate!(cache,f::ModifiedModalC0Basis{D,T,V},x::AbstractVector{<:Point}) where {D,T,V}
  r, v, c = cache
  np = length(x)
  ndof = length(f.terms)*num_components(T)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _evaluate_nd_mmc0!(v,xi,f.ξ₀,f.ξ₁,f.orders,f.terms,c)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{1,ModifiedModalC0Basis{D,V,W}},
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
  fg::FieldGradientArray{1,ModifiedModalC0Basis{D,T,V}},
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
    _gradient_nd_mmc0!(v,xi,f.ξ₀,f.ξ₁,f.orders,f.terms,c,g,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function return_cache(
  fg::FieldGradientArray{2,ModifiedModalC0Basis{D,V,W}},
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
  fg::FieldGradientArray{2,ModifiedModalC0Basis{D,T,V}},
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
    _hessian_nd_mmc0!(v,xi,f.ξ₀,f.ξ₁,f.orders,f.terms,c,g,h,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

# Helpers

function _evaluate_1d_mmc0!(v::AbstractMatrix{T},x,ξ₀,ξ₁,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = ( ξ₁[d] - x[d] ) / ( ξ₁[d] - ξ₀[d] )
  @inbounds v[d,2] = ( x[d] - ξ₀[d] ) / ( ξ₁[d] - ξ₀[d] )
  if n > 2
    ξ = -1 + 2*x[d]
    for i in 3:n
      @inbounds v[d,i] = -sqrt(2*i-3)*v[d,1]*v[d,2]*jacobi(ξ,i-3,1,1)/(i-2)
    end
  end
end

function _gradient_1d_mmc0!(v::AbstractMatrix{T},x,ξ₀,ξ₁,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = -1 / ( ξ₁[d] - ξ₀[d] )
  @inbounds v[d,2] = 1 / ( ξ₁[d] - ξ₀[d] )
  if n > 2
    ξ = -1 + 2*x[d]
    v1 = ( ξ₁[d] - x[d] ) / ( ξ₁[d] - ξ₀[d] )
    v2 = ( x[d] - ξ₀[d] ) / ( ξ₁[d] - ξ₀[d] )
    for i in 3:n
      j, dj = jacobi_and_derivative(ξ,i-3,1,1)
      @inbounds v[d,i] = -sqrt(2*i-3)*(v[d,1]*v2*j+v1*v[d,2]*j+v1*v2*2*dj)/(i-2)
    end
  end
end

function _hessian_1d_mmc0!(v::AbstractMatrix{T},x,ξ₀,ξ₁,order,d) where T
  @assert order > 0
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = 0
  @inbounds v[d,2] = 0
  if n > 2
    ξ = -1 + 2*x[d]
    v1 = ( ξ₁[d] - x[d] ) / ( ξ₁[d] - ξ₀[d] )
    v2 = ( x[d] - ξ₀[d] ) / ( ξ₁[d] - ξ₀[d] )
    dv1 = -1 / ( ξ₁[d] - ξ₀[d] )
    dv2 = 1 / ( ξ₁[d] - ξ₀[d] )
    for i in 3:n
      j, dj = jacobi_and_derivative(ξ,i-3,1,1)
      _, d2j = jacobi_and_derivative(ξ,i-4,2,2)
      @inbounds v[d,i] = -sqrt(2*i-3)*(2*dv1*dv2*j+2*(dv1*v2+v1*dv2)*2*dj+v1*v2*d2j*2*i)/(i-2)
    end
  end
end

function _evaluate_nd_mmc0!(
  v::AbstractVector{V},
  x,
  ξ₀,
  ξ₁,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d_mmc0!(c,x,ξ₀,ξ₁,orders[d],d)
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

function _gradient_nd_mmc0!(
  v::AbstractVector{G},
  x,
  ξ₀,
  ξ₁,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d_mmc0!(c,x,ξ₀,ξ₁,orders[d],d)
    _gradient_1d_mmc0!(g,x,ξ₀,ξ₁,orders[d],d)
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

function _hessian_nd_mmc0!(
  v::AbstractVector{G},
  x,
  ξ₀,
  ξ₁,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  h::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d_mmc0!(c,x,ξ₀,ξ₁,orders[d],d)
    _gradient_1d_mmc0!(g,x,ξ₀,ξ₁,orders[d],d)
    _hessian_1d_mmc0!(h,x,ξ₀,ξ₁,orders[d],d)
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
