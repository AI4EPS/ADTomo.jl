push!(LOAD_PATH,"../src")
using ADCME
using ADTomo
using PyCall
using CSV
using DataFrames
using HDF5

folder = "readin_data/3/"; prange = 2; srange = 5
rfile = open(folder * "range.txt","r")
m = parse(Int16,readline(rfile)); n = parse(Int16,readline(rfile))
l = parse(Int16,readline(rfile)); h = parse(Float16,readline(rfile))
pvs = parse(Float64,readline(rfile))

allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)

fvel0_p = h5read(folder * "1D_fvel0_p.h5","data")
fvel0_s = h5read(folder * "1D_fvel0_s.h5","data")
scaltime_p = h5read(folder * "for_P/scaltime_p.h5","matrix")
scaltime_s = h5read(folder * "for_S/scaltime_s.h5","matrix")
uobs_p = h5read(folder * "for_P/uobs_p.h5","matrix")
uobs_s = h5read(folder * "for_S/uobs_s.h5","matrix")

len = 10; fvel_p = ones(m,n,l); fvel_s = ones(m,n,l)
for i = 0:m-1
    for j = 0:n-1
        for k = 0:l-1
            ii = (i - i%len)/len
            jj = (j - j%len)/len
            kk = (k - k%len)/len
            if (ii + jj + kk)%2 == 0
                fvel_p[i+1,j+1,k+1] = fvel0_p[i+1,j+1,k+1] / 1.2
                fvel_s[i+1,j+1,k+1] = fvel0_s[i+1,j+1,k+1] / 1.2
            else
                fvel_p[i+1,j+1,k+1] = fvel0_p[i+1,j+1,k+1] / 0.8
                fvel_s[i+1,j+1,k+1] = fvel0_s[i+1,j+1,k+1] / 0.8
            end
        end
    end
end

u_p = PyObject[]; u_s = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    
    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_p[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_p[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_p[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_p[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_p[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_p[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_p[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_p[ixd,iyd,izd]
    push!(u_p,eikonal3d(u0,fvel_p,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_s[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_s[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_s[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_s[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_s[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_s[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_s[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_s[ixd,iyd,izd]
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

h5write(folder * "for_P/ucheck_p.h5","matrix",ucheck_p)
h5write(folder * "for_S/ucheck_s.h5","matrix",ucheck_s)
