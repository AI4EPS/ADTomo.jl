using CSV
using ADCME
using DataFrames
using Serialization
using HDF5
using PyCall
using Dates
using PyPlot
using Colors
using JSON
using PyCall

region = "demo/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile("../local/" * region * "readin_data/config.json")["post_rect"]

rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

lambda = config["lambda_p"]
vel0 = h5read(folder * "velocity/vel0_p.h5","data")
folder = folder * "inv_P_"*string(lambda)

sess = Session(); init(sess)
if !isdir(folder * "/plot/")
    mkdir(folder * "/plot/")
    mkdir(folder * "/post/")
end
for ite =10:10:100
    #
    fvar = h5read(folder * "/intermediate/iter_$ite.h5","data")
    fvar = tf.reshape(fvar,(m,n,l)); fvel = run(sess,fvar)
    vel = ones(m,n,l)
    for i = 1:m
        for j = 1:n
            for k = 1:l
                vel[i,j,k] = 2*sigmoid(fvel[i,j,k])-1 + vel0[i,j,k]
            end
        end
    end
    #vel = h5read(folder * "post_$ite.h5","data")
    figure(figsize = (config["width"],config["length"]))
    for i = 1:16
        subplot(4,4,i)
        
        #pcolormesh(plotf2,cmap = "Spectral",vmin=minimum(plotf2),vmax=maximum(plotf2))
        pcolormesh(transpose(vel[:,:,i]), cmap = "seismic",vmin=vel[1,1,i]-1,vmax=vel[1,1,i]+1)
        #pcolormesh(plotf2,cmap = "seismic", vmin = vref*0.7, vmax = vref*1.3)

        title("layer "*string(i))
        colorbar()
    end
    savefig(folder * "/plot/plot_$ite.png")
    close()
    if !isfile(folder * "/post/post_$ite.h5")
        h5write(folder * "/post/post_$ite.h5","data",vel)
    end
end