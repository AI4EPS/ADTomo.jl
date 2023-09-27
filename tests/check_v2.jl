push!(LOAD_PATH,"../src")
using ADCME
using ADTomo
using PyCall
using CSV
using DataFrames
using HDF5

prange = 1.5; srange = 3
rfile = open("readin_data/range.txt", "r")
m = parse(Int16,readline(rfile)); n = parse(Int16,readline(rfile))
l = parse(Int16,readline(rfile)); h = parse(Float16,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

folder = "readin_data/sta_eve/cluster_new2/"
allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)
folder = "readin_data/velocity/vel_2/"
vel0_p = h5read(folder * "vel0_p.h5","data")
vel0_s = h5read(folder * "vel0_s.h5","data")
folder = "readin_data/store/new2/2/"
uobs_p = h5read(folder * "for_P/uobs_p.h5","matrix")
uobs_s = h5read(folder * "for_S/uobs_s.h5","matrix")

len = 10
for i = 0:m-1
    for j = 0:n-1
        for k = 0:l-1
            ii = (i-i%len)/len
            jj = (j-j%len)/len
            kk = (k-k%len)/len
            if (ii+jj+kk)%2 ==0
                vel0_p[i+1,j+1,k+1] = vel0_p[i+1,j+1,k+1] * 1.2
                vel0_s[i+1,j+1,k+1] = vel0_s[i+1,j+1,k+1] * 1.2
            else
                vel0_p[i+1,j+1,k+1] = vel0_p[i+1,j+1,k+1] * 0.8
                vel0_s[i+1,j+1,k+1] = vel0_s[i+1,j+1,k+1] * 0.8
            end
        end
    end
end
fvel_p = 1 ./ vel0_p; fvel_s = 1 ./ vel0_s
folder = "readin_data/velocity/vel_2/"
h5write(folder * "vel_check_p_10.h5","data",vel0_p)
h5write(folder * "vel_check_s_10.h5","data",vel0_s)

u_p = PyObject[]; u_s = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    slowness_pu = fvel_p[ixu,iyu,izu]; slowness_pd = fvel_p[ixu,iyu,izd]
    slowness_su = fvel_s[ixu,iyu,izu]; slowness_sd = fvel_s[ixu,iyu,izd]

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    push!(u_p,eikonal3d(u0,fvel_p,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    push!(u_s,eikonal3d(u0,fvel_s,h,m,n,l,1e-3,false))
end

sess = Session(); init(sess)
u1_p = run(sess,u_p); u1_s = run(sess,u_s)
caltime_p = ones(numsta,numeve); caltime_s = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        
        # P wave
        if x1 == x2
            tx11 = u1_p[i][x1,y1,z1]; tx12 = u1_p[i][x1,y1,z2]
            tx21 = u1_p[i][x1,y2,z1]; tx22 = u1_p[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*u1_p[i][x1,y1,z1] + (jx-x1)*u1_p[i][x2,y1,z1]
            tx12 = (x2-jx)*u1_p[i][x1,y1,z2] + (jx-x1)*u1_p[i][x2,y1,z2]
            tx21 = (x2-jx)*u1_p[i][x1,y2,z1] + (jx-x1)*u1_p[i][x2,y2,z1]
            tx22 = (x2-jx)*u1_p[i][x1,y2,z2] + (jx-x1)*u1_p[i][x2,y2,z2]
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
        caltime_p[i,j] = txyz
        # S wave
        if x1 == x2
            tx11 = u1_s[i][x1,y1,z1]; tx12 = u1_s[i][x1,y1,z2]
            tx21 = u1_s[i][x1,y2,z1]; tx22 = u1_s[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*u1_s[i][x1,y1,z1] + (jx-x1)*u1_s[i][x2,y1,z1]
            tx12 = (x2-jx)*u1_s[i][x1,y1,z2] + (jx-x1)*u1_s[i][x2,y1,z2]
            tx21 = (x2-jx)*u1_s[i][x1,y2,z1] + (jx-x1)*u1_s[i][x2,y2,z1]
            tx22 = (x2-jx)*u1_s[i][x1,y2,z2] + (jx-x1)*u1_s[i][x2,y2,z2]
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
        caltime_s[i,j] = txyz
    end
end

ucheck_p = -ones(numsta,numeve); ucheck_s = -ones(numsta,numeve)
for i = 1:numeve
    for j = 1:numsta
        if uobs_p[j,i] == -1
            continue
        end
        ucheck_p[j,i] = caltime_p[j,i]
    end
    for j = 1:numsta
        if uobs_s[j,i] == -1
            continue
        end
        ucheck_s[j,i] = caltime_s[j,i]    
    end
end

folder = "readin_data/store/new2/2/"
h5write(folder * "for_P/ucheck_p_10.h5","matrix",ucheck_p)
h5write(folder * "for_S/ucheck_s_10.h5","matrix",ucheck_s)