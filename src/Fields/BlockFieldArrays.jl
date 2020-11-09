
function similar_range(r::Base.OneTo,n::Integer)
  Base.OneTo(Int(n))
end

function similar_range(r::BlockedUnitRange,n::Integer)
  blockedrange([n])
end

function similar_range(r::MultiLevelBlockedUnitRange,n::Integer)
  r = similar_range(first(r.local_ranges),n)
  append_ranges([r])
end

# Restricted for a single non zero block
struct BlockFieldArrayCoo{T,N,A,X} <: AbstractBlockArray{T,N}
  axes::X
  blockids::Vector{NTuple{N,Int}}
  block::A
  function BlockFieldArrayCoo(
    _axes::NTuple{N},
    blockids::Vector{NTuple{N,Int}},
    block::A) where {T,N,A<:AbstractArray{T,N}}

    @assert length(blockids) == 1
    #I = first(blockids)
    #@check blocks_equal(axes(block),map(local_range,_axes,I)) "The given block and axes are incompatible."

    X = typeof(_axes)
    new{T,N,A,X}(_axes,blockids,block)
  end
end

struct BlockFieldArrayCooMap <: Map end

function evaluate!(cache,k::BlockFieldArrayCooMap,axes,blockids,block)
  BlockFieldArrayCoo(axes,blockids,block)
end

testitem(a::BlockFieldArrayCoo) = testitem(a.block)

# Specific API

function is_zero_block(a::BlockFieldArrayCoo{T,N},i::Vararg{Integer,N}) where {T,N}
  i != first(a.blockids)
end

# AbstractBlockArray

@inline function BlockArrays.getblock(a::BlockFieldArrayCoo{T,N}, i::Vararg{Integer, N}) where {T,N}
  if i == first(a.blockids)
    a.block
  else
    @notimplemented "Cannot get a zero block from a BlockFieldArrayCoo"
  end
end

# AbstractArray

Base.size(a::BlockFieldArrayCoo) = map(length,Base.axes(a))
Base.axes(a::BlockFieldArrayCoo) = a.axes
Base.IndexStyle(::Type{<:BlockFieldArrayCoo}) = IndexCartesian()

function Base.getindex(a::BlockFieldArrayCoo{T,N},i::Vararg{Integer,N}) where {T,N}
  s = map(findblockindex,a.axes,i)
  ai = a[s...]
  ai
end

# Evaluation

function return_cache(a::BlockFieldArrayCoo,x::Point)
  fc = return_cache(a.block,x)
  fx = return_value(a.block,x)
  cr = return_cache(BlockArrayCooMap(),a.axes,a.blockids,fx)
  (fc,cr)
end

@inline function evaluate!(cache,a::BlockFieldArrayCoo,x::Point)
  fc,cr = cache
  fx = evaluate!(fc,a.block,x)
  evaluate!(cr,BlockArrayCooMap(),a.axes,a.blockids,fx)
end

function return_cache(a::BlockFieldArrayCoo,x::AbstractVector{<:Point})
  fc = return_cache(a.block,x)
  fx = return_value(a.block,x)
  pr = similar_range(first(a.axes),length(x))
  axs = (pr,a.axes...)
  blockids = map(i->(1,i...),a.blockids)
  cr = return_cache(BlockArrayCooMap(),axs,blockids,fx)
  (fc,axs,blockids,cr)
end

@inline function evaluate!(cache,a::BlockFieldArrayCoo,x::AbstractVector{<:Point})
  fc,axs,blockids,cr = cache
  fx = evaluate!(fc,a.block,x)
  evaluate!(cr,BlockArrayCooMap(),axs,blockids,fx)
end

# Gradient

function evaluate!(cache,k::Broadcasting{typeof(∇)},a::BlockFieldArrayCoo)
  g = k(a.block)
  BlockFieldArrayCoo(a.axes,a.blockids,g)
end

function evaluate!(cache,k::Broadcasting{typeof(∇∇)},a::BlockFieldArrayCoo)
  g = k(a.block)
  BlockFieldArrayCoo(a.axes,a.blockids,g)
end

# Transpose

function Base.transpose(a::BlockFieldArrayCoo{T,1} where T)
  r = similar_range(first(axes(a)),1)
  axs = (r,axes(a)...)
  blockids = map(i->(1,i...),a.blockids)
  BlockFieldArrayCoo(axs,blockids,transpose(a.block))
end

# Global optimizations

function lazy_map(::typeof(evaluate),a::LazyArray{<:Fill{}})
end






# Operations

#function return_cache(k::Broadcasting,a::BlockArrayCoo)
#  r = BlockArrayCoo(a.axes,a.blockids,map(k,a.blocks))
#  CachedArray(r)
#end
#
#function evaluate!(cache,k::Broadcasting,a::BlockArrayCoo)
#  setaxes!(cache,a.axes)
#  r = cache.array
#end




## The following masted types are needed to achieve type-stability
## in order to use BlockVectorCoo with arrays of fields
#
#struct MaskedField{F} <: Field
#  field::F
#  mask::Bool
#end
#
#function return_cache(z::MaskedField,x::Point)
#  return_cache(z.field,x)
#end
#
#@inline function evaluate!(cache,z::MaskedField,x::Point) 
#  fx = evaluate!(cache,z.field,x)
#  z.mask ? zero(fx) : fx
#end
#
#testvalue(::Type{MaskedField{F}}) where F = MaskedField(testvalue(F),false)
#
#function return_cache(z::MaskedField,x::AbstractArray{<:Point})
#  return_cache(z.field,x)
#end
#
#function evaluate!(c,z::MaskedField,x::AbstractArray{<:Point})
#  fx = evaluate!(c,z.field,x)
#  if z.mask
#    fill!(fx,zero(eltype(fx)))
#  end
#  fx
#end
#
#@inline gradient(z::MaskedField) = MaskedField(gradient(z.field),z.mask)
#
#struct MaskedFieldArray{T,N,A,X} <: AbstractArray{T,N}
#  axes::X
#  field_array::A
#  mask::Bool
#  function MaskedFieldArray(
#    _axes::NTuple{N},
#    field_array::AbstractArray{F,N},
#    mask::Bool) where {F<:Field,N}
#
#    @check mask || blocks_equal(_axes,axes(field_array)) "Incompatible axes and field_array"
#
#    T = MaskedField{F}
#    A = typeof(field_array)
#    X = typeof(_axes)
#    new{T,N,A,X}(_axes,field_array,mask)
#  end
#end
#
#Base.size(a::MaskedFieldArray) = map(length,Base.axes(a))
#Base.axes(a::MaskedFieldArray) = a.axes
#Base.IndexStyle(::Type{<:MaskedFieldArray}) = IndexCartesian()
#function Base.getindex(a::MaskedFieldArray{T,N},i::Vararg{Integer,N}) where {T,N}
#  if a.mask
#    MaskedField(testitem(a.field_array),a.mask)
#  else
#    MaskedField(a.field_array[i...],a.mask)
#  end
#end
#
##function return_cache(a::MaskedFieldArray,x::Point)
##  cf = return_cache(a.field_array,x)
##  fx = return_value(a.field_array,x)
##  cz = CachedArray(similar(fx,eltype(fx),a.axes))
##  (cf,cz)
##end
##
##@inline function evaluate!(cache,a::MaskedFieldArray,x::Point)
##  cf, cz = cache
##  if a.mask
##    setaxes!(cz,a.axes)
##    r = cz.array
##    fill_entries!(r,zero(eltype(r)))
##    r
##  else
##    evaluate!(cf,a.field_array,x)
##  end
##end
##
##function return_cache(a::MaskedFieldArray,x::AbstractVector{<:Point})
##  cf = return_cache(a.field_array,x)
##  fx = return_value(a.field_array,x)
##  rx = similar_range(first(a.axes),length(x))
##  shape = (,)
##  cz = CachedArray(similar(fx,eltype(fx),a.axes))
##  (cf,cz)
##end
##
##@inline function evaluate!(cache,a::MaskedFieldArray,x::AbstractVector{<:Point})
##  cf, cz = cache
##  if a.mask
##    setaxes!(cz,a.axes)
##    r = cz.array
##    fill_entries!(r,zero(eltype(r)))
##    r
##  else
##    evaluate!(cf,a.field_array,x)
##  end
##end

#function _x_range(x:AbstractVector{<:Point},ran::Tuple{Base.OneTo})
#  Base.OneTo(length(x))
#end
#
#function _x_range(x:AbstractVector{<:Point},ran::Tuple{BlockedUnitRange})
#  blockedrange([length(x)])
#end
#
#function _x_range(x:AbstractVector{<:Point},ran::Tuple{BlockedUnitRange})
#_x_range(x:AbstractVector{<:Point},ran::Tuple{BlockedUnitRange})
#  blockedrange([length(x)])
#end
#  blockedrange([length(x)])
#end
#
#function _axes_with_x(x:AbstractVector{<:Point},ran::NTuple{N,<:BlockedUnitRange} where N)
#  np = length(x)
#  (blockedrange([np]),ran...)
#end
#
#function _new_axes(x,ran::NTuple{N,<:MultiLevelBlockedUnitRange} where N)
#  np = length(x)
#  a = _new_axes(x,first())
#  r = blockedrange([np])
#  (blockedrange([r]),ran...)
#end


#
#@inline function evaluate!(cache,a::BlockFieldArray,x::Point)
#  fc, blockids, cr = cache
#  fx = evaluate!(fc,a.field_array,x)
#  blockids[1] = a.blockid
#  evaluate!(cr,BlockVectorCoo,a.axes,blockids,fx)
#end


#struct BlockFieldArray{T,N,A,X} <: AbstractBlockArray{T,N}
#  axes::X
#  blockid::NTuple{N,Int}
#  field_array::A
#  function BlockFieldArray(
#    axes::NTuple{N},
#    blockid::NTuple{N,Int},
#    field_array::AbstractArray{T,N}) where {T<:Field,N}
#
#    @check begin 
#      msg = "The given field_array and axes are incompatible."
#      blocks_equal(axes(field_array),map(local_range,axes,blockid)) msg
#    end
#
#    A = typeof(field_array)
#    X = typeof(axes)
#    new{T,N,A,X}(axes,blockid,field_array)
#  end
#end
#
#Base.size(a::BlockFieldArray) = map(length,Base.axes(a))
#Base.axes(a::BlockFieldArray) = a.axes
#Base.IndexStyle(::Type{<:BlockFieldArray}) = IndexCartesian()
#Base.getindex(a::BlockFieldArray{T,N},i::Vararg{Integer,N}) where {T,N} = @notimplemented
#Base.setindex!(a::BlockFieldArray{T,N},v,i::Vararg{Integer,N}) where {T,N} = @notimplemented
#
#is_zero_block(a::BlockFieldArray{T,N},i::Vararg{Integer,N}) where {T,N} = i!=a.blockid
#BlockArrays.eachblock(a::BlockFieldArray) = ( a[I]  for I in eachblockid(a) )
#function BlockArrays.getblock(a::BlockFieldArray{T,N}, block::Vararg{Integer, N}) where {T,N}
#  if block == a.blockid
#    a.field_array
#  else
#    ai = testitem(a.field_array)
#    laxes = 
#    fill(zero(a),)
#
#  end
#end
#
#function return_cache(a::BlockFieldArray,x::Point)
#  fc = return_cache(a.field_array,x)
#  fx = return_value(a.field_array,x)
#  blockids = [a.blockid]
#  cr = return_cache(BlockVectorCoo,a.axes,blockids,fx)
#  (fc,blockids,cr)
#end
#
#@inline function evaluate!(cache,a::BlockFieldArray,x::Point)
#  fc, blockids, cr = cache
#  fx = evaluate!(fc,a.field_array,x)
#  blockids[1] = a.blockid
#  evaluate!(cr,BlockVectorCoo,a.axes,blockids,fx)
#end
