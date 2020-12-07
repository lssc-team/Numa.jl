module ModifiedModalC0BasesTests

using Test
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Polynomials
# using BenchmarkTools

import Gridap.Fields: Broadcasting

# Real-valued Q space with isotropic order

x1 = Point(0.0)
x2 = Point(0.5)
x3 = Point(1.0)

V = Float64
G = gradient_type(V,x1)
H = gradient_type(G,x1)
order = 12

b1 = ModifiedModalC0Basis{1}(V,order)
∇b1 = Broadcasting(∇)(b1)
∇∇b1 = Broadcasting(∇)(∇b1)

b2 = ModalC0Basis{1}(V,order)
∇b2 = Broadcasting(∇)(b2)
∇∇b2 = Broadcasting(∇)(∇b2)

@test evaluate(b1,[x1,x2,x3,]) ≈ evaluate(b2,[x1,x2,x3,])
@test evaluate(∇b1,[x1,x2,x3,]) ≈ evaluate(∇b2,[x1,x2,x3,])
@test evaluate(∇∇b1,[x1,x2,x3,]) ≈ evaluate(∇∇b2,[x1,x2,x3,])

end # module
