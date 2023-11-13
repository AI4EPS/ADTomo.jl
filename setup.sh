#!/bin/bash
wget -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz 
tar -xzvf julia.tar.gz
echo "export PATH=\$PATH:$PWD/julia-1.9.3/bin" >> ~/.bashrc
echo "export PATH=\$PATH:$PWD/julia-1.9.3/bin" >> ~/.zshrc

$PWD/julia-1.9.3/bin/julia -e 'using Pkg; Pkg.add(["ADCME"])'
$PWD/julia-1.9.3/bin/julia -e 'using Pkg; Pkg.add(["PyCall", "PyPlot", "CSV", "LinearAlgebra", "DataFrames", "HDF5", "Dates", "LineSearches", "Random", "Optim", "JSON", "Clustering", "Distances"])'
$PWD/julia-1.9.3/bin/julia -e 'using ADCME; include("deps/build.jl")'