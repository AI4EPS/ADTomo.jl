module ADTomo
    
    using ADCME
    using Optim
    using PyCall
    using PyPlot
    using Printf
    using HDF5

    np = PyNULL()
    function __init__()
        copy!(np, pyimport("numpy"))
    end

    include("eikonal_op.jl")
    include("mpi_optimize.jl")

end
