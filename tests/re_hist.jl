push!(LOAD_PATH,"../src")
using CSV
using DataFrames
using HDF5
using ADCME
using ADTomo
using PyCall
using Dates
using PyPlot
using JSON
using Random
using Optim
using LineSearches
Random.seed!(233)

rfile = open("readin_data/range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

folder = "readin_data/sta_eve/cluster_new4/"
allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)

folder = "readin_data/store/new4/2/"
uobs = h5read(folder * "for_P/uobs_p.h5","matrix")
qua = h5read(folder * "for_P/qua_p.h5","matrix")

vel0 = h5read("readin_data/store/new4/2/inv_P_0.05/intermediate/post_15.h5","data")
#vel0 = h5read("/home/lingxia/ADTomo.jl/tests/readin_data/velocity/vel_2/vel0_p.h5","data")
uvar = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    slowness_u = 1/vel0[ixu,iyu,izu]; slowness_d = 1/vel0[ixu,iyu,izd]
    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixd,iyd,izd]
    push!(uvar,eikonal3d(u0,1 ./ vel0,h,m,n,l,1e-3,false))
end

sess = Session(); init(sess)
ucal = run(sess,uvar); caltime = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))

        if x1 == x2
            tx11 = ucal[i][x1,y1,z1]; tx12 = ucal[i][x1,y1,z2]
            tx21 = ucal[i][x1,y2,z1]; tx22 = ucal[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*ucal[i][x1,y1,z1] + (jx-x1)*ucal[i][x2,y1,z1]
            tx12 = (x2-jx)*ucal[i][x1,y1,z2] + (jx-x1)*ucal[i][x2,y1,z2]
            tx21 = (x2-jx)*ucal[i][x1,y2,z1] + (jx-x1)*ucal[i][x2,y2,z1]
            tx22 = (x2-jx)*ucal[i][x1,y2,z2] + (jx-x1)*ucal[i][x2,y2,z2]
        end
        if y1 == y2
            txy1 = tx11; txy2 = tx12
        else
            txy1 = (y2-jy)*tx11 + (jy-y1)*tx21
            txy2 = (y2-jy)*tx12 + (jy-y1)*tx22
        end
        if z1 ==z2
            txyz = txy1
        else
            txyz = (z2-jz)*txy1 + (jz-z1)*txy2
        end
        caltime[i,j] = txyz
    end
end

delt = []; sum_res = 0
for i = 1:numsta
    for j = 1:numeve
        if uobs[i,j] != -1
            push!(delt,uobs[i,j]-caltime[i,j])
            global sum_res += qua[i,j] * (uobs[i,j]-caltime[i,j])^2
        end
    end
end

folder = "readin_data/store/new4/2/inv_P_0.05/"
plt.figure(); plt.hist(delt,bins=60,edgecolor="royalblue",color="skyblue");
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-1.5,1.5)
plt.title("Histogram_P");plt.savefig(folder * "hist_p_15.png")
@show sum_res