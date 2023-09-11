# ADTomo

## Install

### install julia
    1.julia wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
    2.tar -xzvf julia-1.9.3-linux-x86_64.tar.gz
    3. vi ~/.bashrc
        add " export PATH="/home/lingxia/julia-1.9.3/bin:$PATH" " to ~/.bashrc
    4. source ~/.bashrc
### install useful packages
    using Pkg
    Pkg.add("ADCME") # a precompile() is required
    Pkg.add("PyCall")
    Pkg.add("PyPlot")
    Pkg.add("CSV")
    Pkg.add("LinearAlgebra")
    Pkg.add("DataFrames")
    Pkg.add("HDF5")
    Pkg.add("Dates")
    Pkg.add("LineSearches")
    Pkg.add("Random")
    Pkg.add("Optim)
### install ADTomo
    1.git clone 
    2.in julia,use
        using ADCME
        include("deps/build.jl")
### install MPI
    in julia 
        using ADCME
        install_openmpi()
    add " export PATH="/home/lingxia/.julia/adcme/bin:$PATH" " to ~/.bashrc
    in julia
        using ADCME
        ADCME.precompile(true)
## Usage