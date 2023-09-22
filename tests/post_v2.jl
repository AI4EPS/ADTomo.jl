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
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile));l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))
folder = "readin_data/velocity/vel_2/"
v0 = h5read(folder * "vel0_p.h5","data")

folder = "readin_data/store/all/2/intermediate/"
for ite = 5:5:300
    fvar = h5read(folder * "iter_$ite.h5","data")
    fvar = tf.reshape(fvar,(m,n,l))
    sess = Session(); init(sess)
    fplot = run(sess,fvar)
    
    figure(figsize = (36,21))
    for i = 1:l
        vref = v0[1,1,i]
        subplot(5,5,i)
        plotf2 = ones(m,n)
        for j = 1:m
            for k = 1:n
                plotf2[j,k] = 2 * sigmoid(fplot[j,k,i]) - 1 + vref
            end
        end
        
        pcolormesh(plotf2,cmap = "seismic",vmin=v0[1,1,i] * 0.8,vmax=v0[1,1,i]*1.2)
        title("layer "*string(i))
        colorbar()
    end
    savefig(folder * "plot_iter_$ite.png")
    close()
end