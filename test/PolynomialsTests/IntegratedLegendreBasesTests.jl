module IntegratedLegendreBasesTests

using Test
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials
# using BenchmarkTools

import Gridap.Polynomials: _legendre
import Gridap.Fields: Broadcasting

@test all(map(i->map(x->_legendre(x,Val{i}()),(-1,1))==(1-2*(i%2),1),0:4))

# REMARK: In all test cases below
# _sort_by_tensor_prod! == _sort_by_n_faces!

# Real-valued Q space with isotropic order

xi = Point(0.5)
V = Float64
G = gradient_type(V,xi)
H = gradient_type(G,xi)
order = 3

b = IntegratedLegendreBasis{1}(V,order)
∇b = Broadcasting(∇)(b)
∇∇b = Broadcasting(∇)(∇b)

v = V[0.5  0.5  -sqrt(3)/4 0.0]
g = G[-1.0 1.0 0.0 -sqrt(5)/2]
h = H[0.0 0.0 2*sqrt(3) 0.0]

# cache = return_cache(b,[xi,])
# @btime evaluate!($cache,$b,$[xi,])
# cache = return_cache(∇b,[xi,])
# @btime evaluate!($cache,$∇b,$[xi,])
# cache = return_cache(∇∇b,[xi,])
# @btime evaluate!($cache,$∇∇b,$[xi,])

@test evaluate(b,[xi,]) ≈ v
@test evaluate(∇b,[xi,]) ≈ g
@test evaluate(∇∇b,[xi,]) ≈ h

# Real-valued Q space with isotropic order

xi = Point(0.5,0.5)
np = 3
x = fill(xi,np)

order = 1
V = Float64
G = gradient_type(V,xi)
H = gradient_type(G,xi)
b = IntegratedLegendreBasis{2}(V,order)

v = V[0.25, 0.25, 0.25, 0.25]
g = G[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
h = H[(0.0, 1.0, 1.0, 0.0), (0.0, -1.0, -1.0, 0.0), (0.0, -1.0, -1.0, 0.0), (0.0, 1.0, 1.0, 0.0)]

bx = repeat(permutedims(v),np)
∇bx = repeat(permutedims(g),np)
Hbx = repeat(permutedims(h),np)
test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# Real-valued Q space with anisotropic order

orders = (1,2)
V = Float64
G = gradient_type(V,xi)
H = gradient_type(G,xi)

b = IntegratedLegendreBasis{2}(V,orders)
∇b = Broadcasting(∇)(b)
∇∇b = Broadcasting(∇)(∇b)

v = V[0.25 0.25 0.25 0.25 -sqrt(3)/8 -sqrt(3)/8]
g = G[(-0.5, -0.5) (0.5, -0.5) (-0.5, 0.5) (0.5, 0.5) (sqrt(3)/4, 0.0) (-sqrt(3)/4, 0.0)]
h = H[(0.0, 1.0, 1.0, 0.0) (0.0, -1.0, -1.0, 0.0) (0.0, -1.0, -1.0, 0.0) (0.0, 1.0, 1.0, 0.0) (0.0, 0.0, 0.0, sqrt(3)) (0.0, 0.0, 0.0, sqrt(3))]

@test evaluate(b,[xi,]) ≈ v
@test evaluate(∇b,[xi,]) ≈ g
@test evaluate(∇∇b,[xi,]) ≈ h

# Vector-valued Q space with isotropic order

order = 1
V = VectorValue{3,Float64}
G = gradient_type(V,xi)
H = gradient_type(G,xi)
b = IntegratedLegendreBasis{2}(V,order)

v = V[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25],
      [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]

g = G[[-0.5 0.0 0.0; -0.5 0.0 0.0], [0.0 -0.5 0.0; 0.0 -0.5 0.0],
      [0.0 0.0 -0.5; 0.0 0.0 -0.5], [0.5 0.0 0.0; -0.5 0.0 0.0],
      [0.0 0.5 0.0; 0.0 -0.5 0.0], [0.0 0.0 0.5; 0.0 0.0 -0.5],
      [-0.5 0.0 0.0; 0.5 0.0 0.0], [0.0 -0.5 0.0; 0.0 0.5 0.0],
      [0.0 0.0 -0.5; 0.0 0.0 0.5], [0.5 0.0 0.0; 0.5 0.0 0.0],
      [0.0 0.5 0.0; 0.0 0.5 0.0], [0.0 0.0 0.5; 0.0 0.0 0.5]]

h = H[
      (0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0),
      (0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0),
      (0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,0.0),
      (0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0),
      (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0)]

bx = repeat(permutedims(v),np)
∇bx = repeat(permutedims(g),np)
Hbx = repeat(permutedims(h),np)
test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# Degenerated case 1: order 0

# xi = Point(2,3)
# np = 5
# x = fill(xi,np)

# order = 0
# V = Float64
# G = gradient_type(V,xi)
# H = gradient_type(G,xi)
# b = MonomialBasis{2}(V,order)
# @test get_order(b) == 0
# @test get_orders(b) == (0,0)

# v = V[1.0,]
# g = G[(0.0, 0.0),]
# h = H[(0.0, 0.0, 0.0, 0.0),]

# bx = repeat(permutedims(v),np)
# ∇bx = repeat(permutedims(g),np)
# Hbx = repeat(permutedims(h),np)
# test_field_array(b,x,bx,grad=∇bx,gradgrad=Hbx)

# Degenerated case 2: void Point instance

# order = 1
# b = MonomialBasis{0}(VectorValue{2,Float64},order)
# @test evaluate(b,Point{0,Float64}[(),()]) == VectorValue{2,Float64}[(1.0, 0.0) (0.0, 1.0); (1.0, 0.0) (0.0, 1.0)]

end # module
