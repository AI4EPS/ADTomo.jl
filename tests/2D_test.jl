using CSV
using DataFrames
using Serialization
using HDF5
using ADCME
using ADTomo
using PyCall
using Dates
using PyPlot
using Random
using Base
using LinearAlgebra
Random.seed!(233)
reset_default_graph()
     

m = 40           #width
n = 30           #length
h = 1.0          #resolution
     

f = ones(n,m) ./ 6
f[16:20,20:24] .= 1/5   
f[8:14,10:18]  .= 1/7              #design velocity model
     

allsrc = DataFrame(x = [], y = [])
allrcv = DataFrame(x = [], y = [])

for i = 1:30
    push!(allrcv.x,rand(1:m))
    push!(allrcv.y,rand(1:n))
end                                #design the locations of stations

for i = 1:40                        #the number of events
    push!(allsrc.x,rand(1:m))
    push!(allsrc.y,rand(1:n))
end                                 #design the locations of events

numrcv = size(allrcv,1)
numsrc = size(allsrc,1)
     

u = PyObject[]
for i=1:numsrc
    push!(u,eikonal(f,allsrc.x[i],allsrc.y[i],h))
end
sess = Session()
init(sess)
uobs = run(sess, u)                                      #uobs is a list of [numsrc * (m,n)] representing travel time
     

obs_time = Array{Float64}(undef,numsrc,numrcv)
for i = 1:numsrc
    for j = 1:numrcv
        obs_time[i,j] = uobs[i][allrcv.y[j],allrcv.x[j]]
    end
end
     

i = 5                                      #choose a source to plot a traveltime image
figure()
pcolormesh(uobs[i], cmap = "Purples")
colorbar().set_label("traveltime/s")
xlabel("x/km"); ylabel("y/km")
scatter(allsrc.x[i],allsrc.y[i])
savefig("traveltime_$i.png")
     
fvar = Variable(ones(n, m) ./ 6)                          #design an original velocity model for inversion
u = PyObject[]
for i=1:numsrc
    push!(u,eikonal(fvar,allsrc.x[i],allsrc.y[i],h))
end
     

loss = sum([sum((obs_time[i,j] - u[i][allrcv.y[j],allrcv.x[j]])^2) for i = 1:numsrc for j=1:numrcv])
init(sess)
@show run(sess, loss)
BFGS!(sess, loss, 200)                                            #200 means max iteration steps and (0.5,100.0) means velocity change range



figure(figsize=(10, 4))
subplot(121)
pcolormesh( 1 ./ f, vmin = 5,vmax = 7)
colorbar()
title("True")
scatter(allsrc.x,allsrc.y,label="event")
scatter(allrcv.x,allrcv.y,label="station")
legend()
subplot(122)
pcolormesh(1 ./ run(sess,fvar),vmin = 5,vmax = 7)                #vmin & vmax
colorbar()
title("Inverted")
savefig("inversion_result.png")                                      