module ADEikonal
    
    using ADCME
    using Optim
    using PyCall
    using PyPlot
    using Printf

    np = PyNULL()
    function __init__()
        copy!(np, pyimport("numpy"))
    end

    include("eikonal_op.jl")

end
