#This code adds a sigmoid function to control the velocity change.
#the other things all base on inv_3.jl
#it will start from a true loss first, then I will add some smoothing on it
ENV["TF_NUM_INTEROP_THREADS"] = 1
using ADCME
using ADTomo
using PyCall
using PyPlot
using CSV
using LinearAlgebra
using DataFrames
using HDF5
using Dates
using Random
using Optim
using LineSearches
Random.seed!(233)

mpi_init()
rank = mpi_rank()
nproc = mpi_size()

folder = "/home/lingxia/ADTomo.jl/tests/readin_data/1/"; prange = 2; srange = 5
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
rpvs = parse(Float64,readline(rfile)); bins = parse(Int,readline(rfile))

allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)
fvel0_p = h5read(folder * "1D_fvel0_p.h5","data")
vel0_p = h5read(folder * "1D_vel0_p.h5","data")
scaltime_p = h5read(folder * "for_P/scaltime_p.h5","matrix")
uobs_p = h5read(folder * "for_P/ucheck_1_p.h5","matrix")
qua_p = h5read(folder * "for_P/qua_p.h5","matrix")

scaltime_p = scaltime_p[rank+1:nproc:numsta,:]
uobs_p = uobs_p[rank+1:nproc:numsta,:]
qua_p = qua_p[rank+1:nproc:numsta,:]
allsta = allsta[rank+1:nproc:numsta,:]
numsta = size(allsta,1)
@show rank, nproc, numsta

var_change = Variable(zero(vel0_p))
fvar_ = 2*sigmoid(var_change)-1 + vel0_p
fvar = mpi_bcast(fvar_)
uvar_p = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)/vel0_p[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)/vel0_p[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)/vel0_p[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)/vel0_p[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)/vel0_p[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)/vel0_p[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)/vel0_p[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)/vel0_p[ixd,iyd,izd]
    push!(uvar_p,eikonal3d(u0,1 ./fvar,h,m,n,l,1e-3,false))
end
caltime_p = []
for i = 1:numsta
    timei_p = []
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        
        if x1 == x2
            tx11 = uvar_p[i][x1,y1,z1]; tx12 = uvar_p[i][x1,y1,z2]
            tx21 = uvar_p[i][x1,y2,z1]; tx22 = uvar_p[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*uvar_p[i][x1,y1,z1] + (jx-x1)*uvar_p[i][x2,y1,z1]
            tx12 = (x2-jx)*uvar_p[i][x1,y1,z2] + (jx-x1)*uvar_p[i][x2,y1,z2]
            tx21 = (x2-jx)*uvar_p[i][x1,y2,z1] + (jx-x1)*uvar_p[i][x2,y2,z1]
            tx22 = (x2-jx)*uvar_p[i][x1,y2,z2] + (jx-x1)*uvar_p[i][x2,y2,z2]
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
        push!(timei_p,txyz)
    end
    push!(caltime_p,timei_p)
end

sum_loss_time = PyObject[]
for i = 1:numeve
    for j = 1:numsta
        if uobs_p[j,i] == -1
            continue
        end
        if abs(uobs_p[j,i]-scaltime_p[j,i]) > prange
            continue
        end
        push!(sum_loss_time, qua_p[j,i]*(uobs_p[j,i]-caltime_p[j][i])^2)
    end
end

laplace_wei = Array{Float64}(undef, 3, 3, 3)
laplace_wei[:,:,1] = [0 0 0; 0 1 0; 0 0 0]
laplace_wei[:,:,2] = [0 1 0; 1 -6 1; 0 1 0]
laplace_wei[:,:,3] = [0 0 0; 0 1 0; 0 0 0]
filter = tf.constant(laplace_wei,shape=(3,3,3,1,1),dtype=tf.float64)

vel = tf.reshape(fvar,(1,m,n,l,1))
cvel = tf.nn.conv3d(vel,filter,strides = (1,1,1,1,1),padding="VALID")
loss = sum(sum_loss_time) #+ 0.0005 * sum(abs(cvel))

sess = Session(); init(sess)

loss = mpi_sum(loss)
@show run(sess,loss)

options = Optim.Options(iterations = 300)
result = ADTomo.mpi_optimize(sess, loss, method="LBFGS", options = options, 
    loc = folder * "intermediate/check_0/", steps = 15)
if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]
    #h5write(folder * "output/inv_P_3.h5","data",result[1])
end
mpi_finalize()