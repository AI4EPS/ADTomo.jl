using CSV
using ADCME
using DataFrames
using Serialization
using HDF5
using PyCall
using Dates
using PyPlot
using Colors

folder = "readin_data/3/"
h5folder = "intermediate/inv_0/"
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
f0 = h5read(folder * "1D_fvel0_p.h5","data")

colors = [colorant"red", colorant"yellow",colorant"deepskyblue",colorant"blue"]
newcmap = ColorMap(colors)

maxx = 0; minn = 10
for ite = 15:15:300
    fvar = h5read(folder * h5folder * "iter_$ite.h5","data")
    fvar = tf.reshape(fvar,(m,n,l))
    sess = Session(); init(sess)
    fplot = run(sess,fvar)
    
    figure(figsize = (20,20))
    for i = 1:l
        vref = 1/f0[1,1,i]
        subplot(5,5,i)
        plotf2 = ones(m,n)
        for j = 1:m
            for k = 1:n
                plotf2[j,k] = 2 * sigmoid(fplot[j,k,i]) - 1 + vref
                global maxx = max(maxx,plotf2[j,k])
                global minn = min(minn,plotf2[j,k])
            end
        end
        
        #pcolormesh(plotf2,cmap = newcmap,vmin = 3,vmax = 8)
        pcolormesh(plotf2,cmap = "seismic",vmin=1 ./f0[1,1,i] * 0.8,vmax=1 ./f0[1,1,i]*1.2)
        title("layer "*string(i))
        colorbar()
    end
    savefig(folder * h5folder * "plot_iter_$ite.png")
    close()
end
@show maxx
@show minn