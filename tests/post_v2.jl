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

rfile = open("readin_data/range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile)); l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))
folder = "readin_data/velocity/vel_2/"
vel0 = h5read(folder * "vel0_s.h5","data")
#=
vel_raw = h5read("readin_data/store/new4/2/inv_P/step_6/01/intermediate/post_20.h5","data")
vel_r = tf.reshape(vel_raw,(m,n,l))
sess = Session(); init(sess)
vel0 = run(sess,vel_r)
=#
sess = Session(); init(sess)
folder = "readin_data/store/new4/2/check_S_0.005/intermediate/"
# folder = "readin_data/store/new4/2/inv_P/step_7/2/intermediate/"
for ite = 1:1:101
    fvar = h5read(folder * "iter_$ite.h5","data")
    fvar = tf.reshape(fvar,(m,n,l)); fvel = run(sess,fvar)
    vel = ones(m,n,l)
    for i = 1:m
        for j = 1:n
            for k = 1:l
                vel[i,j,k] = 2*sigmoid(fvel[i,j,k])-1 + vel0[i,j,k]
            end
        end
    end

    figure(figsize = (20,40))
    for i = 1:16
        subplot(4,4,i)
        
        #pcolormesh(plotf2,cmap = "Spectral",vmin=minimum(plotf2),vmax=maximum(plotf2))
        pcolormesh(transpose(vel[:,:,i]), cmap = "seismic",vmin=sum(vel[:,:,i])/(m*n)-1,vmax=sum(vel[:,:,i])/(m*n)+1)
        #pcolormesh(plotf2,cmap = "seismic", vmin = vref*0.7, vmax = vref*1.3)

        title("layer "*string(i))
        colorbar()
    end
    savefig(folder * "plot_$ite.png")
    close()
    h5write(folder * "post_$ite.h5","data",vel)
end