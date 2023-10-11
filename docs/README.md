# ADTomo

## Install

### install julia
    1. get julia 
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
    2. unzip
        tar -xzvf julia-1.9.3-linux-x86_64.tar.gz
    3. add path
        vi ~/.bashrc
        add " export PATH="～/julia-1.9.3/bin:$PATH" " to ~/.bashrc
        source ~/.bashrc
### install necessary packages
    ```julia
    using Pkg
    pkgs_to_install = [
        "ADCME",     # 需要运行 precompile()
        "PyCall",
        "PyPlot",
        "CSV",
        "LinearAlgebra",
        "DataFrames",
        "HDF5",
        "Dates",
        "LineSearches",
        "Random",
        "Optim"
    ]
    Pkg.add.(pkgs_to_install)
    ```
### install ADTomo
    1. download the codebase
        git clone 
    2. run the build code
    ```julia
        using ADCME
        include("deps/build.jl")
    ```
### install MPI
    ```julia
        using ADCME
        install_openmpi()
    add path
        vi ~/.bashrc
        add " export PATH="～/.julia/adcme/bin:$PATH" " to ~/.bashrc
        source ~/.bashrc
    ```julia
        using ADCME
        ADCME.precompile(true)
    ```
## Usage

### get the data of seismic wave picks

In this part, we do not pick arrival time manually and use Phasenet to get all traveltime data.

1. choose the region, period, networks and channels by setting "congig.py" 
2. run "download_waveform_event.py" to get all the waveforms in this set region.
3. run "run_phasenet.py" to get all the required data of P picks and S picks in the folder with the name of the region.

### preprocess

In this part, we prepare the locations of stations and events and adjust the 1D velocity model for the inversion part. 
#### get stations and events
    parameters need to be adjusted: 
        resolution: the size of each grid
        rotated theta: the direction compared with the original longitude direction and latitide direction
        P(S) requirement: the smallest number the phase score should reach
        range: the area it will use(like z should below 15km, and other data is filtered) 
        eps, min_pts: DBSCAN parameters
        ratio: if a cluster have 50 events, then we reserve 2, the ratio equals 1/25.
    run the code "sta_eve.jl" we can get 2 files, recording the xyz of events and stations, the number that they are recorded in the original files and the location map of these stations and events.
#### prepare a proper 1D velocity model.
    design a 1D velocity model or just use the common one and adjust it based on the observation. And the input file should be a 3D matrix.
#### get observed traveltime 
    run the "gene_obs.jl" code in this part
    1. from the calculated traveltime to pick observed travel time
    2. generate histogram of residuals and residuals of stations and events on the topography map
    3. plot the coverage of the observed data
    (maybe in this part, from the output of camparing P picks, S picks and the loss function, you can adjust velocity model a bit more)

### inversion
    parameters need to be adjust: the range the initial velocity model can change, Gaussian regularization(size, and lambda)

### post process
    "post_v2.jl" : read the intermediate results and plot it in the form of rectangule
    "post_v3.py" : read the intermediate results and save the information of "lon" and "lat" in the file