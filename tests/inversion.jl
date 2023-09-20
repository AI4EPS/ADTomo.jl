push!(LOAD_PATH,"../src")
ENV["TF_NUM_INTEROP_THREADS"] = 1
using ADCME
using ADTomo
using PyCall
using PyPlot
using CSV
using DataFrames
using HDF5
using LinearAlgebra
using JSON
using Random
using Optim
using LineSearches
Random.seed!(233)

mpi_init()
rank = mpi_rank()
nproc = mpi_size()

prange = 1.5; srange = 3
rfile = open("readin_data/range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

folder = "readin_data/sta_eve/cluster_all/"
allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)
folder = "readin_data/velocity/vel_2/"
vel0 = h5read(folder * "vel0_p.h5","data")
folder = "readin_data/store/all/2/"
uobs = h5read(folder * "for_P/uobs_p.h5","matrix")
qua = h5read(folder * "for_P/qua_p.h5","matrix")

allsta = allsta[rank+1:nproc:numsta,:]
uobs = uobs[rank+1:nproc:numsta,:]
qua = qua[rank+1:nproc:numsta,:]
numsta = size(allsta,1)
@show rank, nproc, numsta
#
var_change = Variable(zero(vel0))
fvar_ = 2*sigmoid(var_change)-1 + vel0
fvar = mpi_bcast(fvar_)
#=
fvel0 = zeros(Float64,l)
for i = 1:l
    fvel0[i] = 1/vel0[1,1,i]
end
fvel = Variable(fvel0)
fvar_ = reshape(repeat(fvel,m*n),(m,n,l))
fvar = mpi_bcast(fvar_)
=#

uvar = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    slowness_u = 1/vel0[ixu,iyu,izu]; slowness_d = 1/vel0[ixu,iyu,izd]
    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_u*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_d*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_u*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_d*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_u*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_d*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_u*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_d*h
    push!(uvar,eikonal3d(u0,1 ./ fvar,h,m,n,l,1e-3,false))
end

caltime = []
for i = 1:numsta
    timei = []
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        
        if x1 == x2
            tx11 = uvar[i][x1,y1,z1]; tx12 = uvar[i][x1,y1,z2]
            tx21 = uvar[i][x1,y2,z1]; tx22 = uvar[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*uvar[i][x1,y1,z1] + (jx-x1)*uvar[i][x2,y1,z1]
            tx12 = (x2-jx)*uvar[i][x1,y1,z2] + (jx-x1)*uvar[i][x2,y1,z2]
            tx21 = (x2-jx)*uvar[i][x1,y2,z1] + (jx-x1)*uvar[i][x2,y2,z1]
            tx22 = (x2-jx)*uvar[i][x1,y2,z2] + (jx-x1)*uvar[i][x2,y2,z2]
        end
        if y1 == y2
            txy1 = tx11; txy2 = tx12
        else
            txy1 = (y2-jy)*tx11 + (jy-y1)*tx21
            txy2 = (y2-jy)*tx12 + (jy-y1)*tx22
        end
        if z1 == z2
            txyz = txy1
        else
            txyz = (z2-jz)*txy1 + (jz-z1)*txy2
        end
        push!(timei,txyz)
    end
    push!(caltime,timei)
end

sum_loss_time = PyObject[]
for i = 1:numeve
    for j = 1:numsta
        if uobs[j,i] == -1
            continue
        end
        push!(sum_loss_time, qua[j,i]*(uobs[j,i]-caltime[j][i])^2)
    end
end

laplace_wei = Array{Float64}(undef, 3, 3, 3)
laplace_wei[:,:,1] = [0 0 0; 0 1 0; 0 0 0]
laplace_wei[:,:,2] = [0 1 0; 1 -6 1; 0 1 0]
laplace_wei[:,:,3] = [0 0 0; 0 1 0; 0 0 0]
filter = tf.constant(laplace_wei,shape=(3,3,3,1,1),dtype=tf.float64)

vel = tf.reshape(fvar,(1,m,n,l,1))
cvel = tf.nn.conv3d(vel,filter,strides = (1,1,1,1,1),padding="VALID")
loss = sum(sum_loss_time) + 0.0005 * sum(abs(cvel))
sess = Session(); init(sess)

loss = mpi_sum(loss)
@show run(sess,loss)

options = Optim.Options(iterations = 300)
result = ADTomo.mpi_optimize(sess, loss, method="LBFGS", options = options, 
    loc = folder * "intermediate/", steps = 5)
if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]
end
mpi_finalize()