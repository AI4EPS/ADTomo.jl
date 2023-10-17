# ADTomo

## Install

### install julia
    1. get julia 
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
    2. unzip
        tar -xzvf julia-1.9.3-linux-x86_64.tar.gz
    3. add path
        vi ~/.bashrc
        add " export PATH="$PATH:/home/usr/julia-1.9.3/bin" " to ~/.bashrc
        source ~/.bashrc
### install necessary packages
    ```julia
    using Pkg
    pkgs_to_install = [
        "ADCME",     ## precompile()
        "PyCall",
        "PyPlot",
        "CSV",
        "LinearAlgebra",
        "DataFrames",
        "HDF5",
        "Dates",
        "LineSearches",
        "Random",
        "Optim",
        "JSON",
        "Clustering",
        "Distances"
    ]
    Pkg.add(pkgs_to_install)
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
        ADCME.precompile(true)
        get_mpirun()
    add path
        vi ~/.bashrc
        add " export PATH="$PATH:home/usr/.julia/adcme/bin" " to ~/.bashrc
        source ~/.bashrc

## Usage

### orders to run the code
1. mkdir local & cd local & download "demo.tar.gz" in the release part to this folder
2. tar -xvf demo.tar.gz
3. cd ../scripts
4. julia set_config.jl            # set parameters and generate "config.json"
5. julia sta_eve.jl
6. julia gene_vel0.jl             # generate GIL7 velocity model
7. julia gene_obs.jl              # you can run it multiple times with changing veltimes_p&veltimes_s in the "config.json" to find an optimal one
8. julia gene_check.jl (optional, generating checkerboard test)
9. mkdir ../local/demo/readin_data/inv_P_0.005 ../local/demo/readin_data/inv_P_0.005/intermediate
\\ (this can help store intermediate results, the name"inv_P_0.005" is base on wave stype and lambda, based on line126-128 in "inversion.jl")
10. mpirun --bind-to core -n 16 julia inversion.jl  # adjust the number of cores based on data
### get the data of seismic wave picks

In this part, we do not pick arrival time manually and use [Phasenet](https://github.com/AI4EPS/PhaseNet) to get all traveltime data.

1. set the region, period, networks and channels by editting "config.json" 
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
    run the code "sta_eve.jl" we can get 2 files, recording the xyz of events and stations,
                                     the number that they are recorded in the original files 
                                     and the location map of these stations and events.
#### prepare a proper 1D velocity model.
    design a 1D velocity model or just use the common one and adjust it based on the observation. And the input file should be a 3D matrix.
#### get observed traveltime 
    run the "gene_obs.jl" code in this part
    1. from the calculated traveltime to pick observed travel time
    2. generate histogram of residuals and residuals of stations and events on the topography map
    3. plot the coverage of the observed data
    (maybe in this part, from the output of camparing P picks, S picks and the loss function, you can adjust velocity model a bit more)
    4. "re_hist.jl" has very similar function as the "gene_obs.jl" code, but much simplier.
### generate checkerboard test
    julia gene_check.jl
    need to decide the length of the grid size, the velocity change in each grid.
    this code will save the velocity of checkerboard test, and the synthetic travel time.
### inversion
    parameters need to be adjust: the range the initial velocity model can change, Gaussian regularization(size, and lambda)


### codes for plotting
#### prepare faults
    1. download the data of fault from USGS
        wget https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip
    2. unzip Qfaults_GIS.zip
    3. download GMT
    4. cd SHP
    5. ogr2ogr -f GMT ca_offshore.gmt ca_offshore.shp
       ogr2ogr -f GMT fault_areas.gmt fault_areas.shp
       ogr2ogr -f GMT Qfaults_US_Database.gmt Qfaults_US_Database.shp
#### prepare the velocity model
    "post_rect.jl": 1. read the intermediate results and plot it in the form of rectangule( the same as inversion part)
                    2. generate ".h5" file of the intermediate and final inversion results which can be used for plotting
    "post_gmt.py" : read the intermediate results and save the information of "lon" and "lat" in the file, which is used in the part of "plot_final.sh"
#### plot with GMT
    "plot_range.sh": need to prepare the lon&lat of the four grids in the corner
    "plot_residual.sh": use the lon&lat of stations and events, as well as a "residual" value to plot the residual befor inversion
    "plot_final.sh": 1. plot information of faults
                     2. plot velocity model from "post_gmt.py"