module PolynomialsTests

using Test

@testset "MonomialBases" begin include("MonomialBasesTests.jl") end

@testset "ModalC0Bases" begin include("ModalC0BasesTests.jl") end

@testset "ModifiedModalC0Bases" begin include("ModifiedModalC0BasesTests.jl") end

@testset "AgFEMModalC0Bases" begin include("AgFEMModalC0BasesTests.jl") end

@testset "QGradMonomialBases" begin include("QGradMonomialBasesTests.jl") end

@testset "QCurlGradMonomialBases" begin include("QCurlGradMonomialBasesTests.jl") end

#@testset "ChangeBasis" begin include("ChangeBasisTests.jl") end

end # module
