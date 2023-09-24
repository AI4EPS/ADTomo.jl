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

prange = 2.5; srange = 5
rfile = open("readin_data/range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

folder = "readin_data/sta_eve/cluster_all/"
stations = CSV.read("seismic_data/BayArea/obspy/stations.csv", DataFrame)
events = CSV.read("seismic_data/BayArea/obspy/catalog.csv", DataFrame)
allsta = CSV.read(folder * "allsta.csv",DataFrame)
alleve = CSV.read(folder * "alleve.csv",DataFrame)
numsta = size(allsta,1); numeve = size(alleve,1)
file = open(folder * "stations.json", "r")
dic_sta = JSON.parse(file); close(file)
eveid = h5read(folder * "eveid.h5","data")
folder = "readin_data/velocity/vel_6/"
vel_p = h5read(folder * "vel0_p.h5","data")
vel_s = h5read(folder * "vel0_s.h5","data")
folder = "readin_data/store/all/6/"

u_p = PyObject[]; u_s = PyObject[]; fvel_p = 1 ./ vel_p; fvel_s = 1 ./ vel_s
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
ubeg_p = run(sess,u_p); ubeg_s = run(sess,u_s)
scaltime_p = ones(numsta,numeve); scaltime_s = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        # P wave
        if x1 == x2
            tx11 = ubeg_p[i][x1,y1,z1]; tx12 = ubeg_p[i][x1,y1,z2]
            tx21 = ubeg_p[i][x1,y2,z1]; tx22 = ubeg_p[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*ubeg_p[i][x1,y1,z1] + (jx-x1)*ubeg_p[i][x2,y1,z1]
            tx12 = (x2-jx)*ubeg_p[i][x1,y1,z2] + (jx-x1)*ubeg_p[i][x2,y1,z2]
            tx21 = (x2-jx)*ubeg_p[i][x1,y2,z1] + (jx-x1)*ubeg_p[i][x2,y2,z1]
            tx22 = (x2-jx)*ubeg_p[i][x1,y2,z2] + (jx-x1)*ubeg_p[i][x2,y2,z2]
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
        scaltime_p[i,j] = txyz
        # S wave
        if x1 == x2
            tx11 = ubeg_s[i][x1,y1,z1]; tx12 = ubeg_s[i][x1,y1,z2]
            tx21 = ubeg_s[i][x1,y2,z1]; tx22 = ubeg_s[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*ubeg_s[i][x1,y1,z1] + (jx-x1)*ubeg_s[i][x2,y1,z1]
            tx12 = (x2-jx)*ubeg_s[i][x1,y1,z2] + (jx-x1)*ubeg_s[i][x2,y1,z2]
            tx21 = (x2-jx)*ubeg_s[i][x1,y2,z1] + (jx-x1)*ubeg_s[i][x2,y2,z1]
            tx22 = (x2-jx)*ubeg_s[i][x1,y2,z2] + (jx-x1)*ubeg_s[i][x2,y2,z2]
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
        scaltime_s[i,j] = txyz
    end
end

uobs_p = -ones(numsta,numeve); uobs_s = -ones(numsta,numeve)
qua_p = ones(numsta,numeve); qua_s = ones(numsta,numeve)

for i = 1:numeve
    local evetime = events[eveid[i],2]; local id = events[eveid[i],1]
    local file = "seismic_data/BayArea/phasenet/picks/" * id * ".csv"
    picks = CSV.read(file,DataFrame)
    numpick = size(picks,1)

    for j = 1:numpick
        sta_idx = dic_sta[picks[j,1]]
        if sta_idx == -1  
            continue   
        end

        delta_militime = picks[j,4] - parse(DateTime,evetime[1:23])
        delta_time = delta_militime.value/1000
        if picks[j,6] == "P" && picks[j,5] > 0.8
            if uobs_p[sta_idx,i] == -1 || abs(delta_time-scaltime_p[sta_idx,i]) < abs(uobs_p[sta_idx,i] - scaltime_p[sta_idx,i])
                uobs_p[sta_idx,i] = delta_time
                qua_p[sta_idx,i] = picks[j,5]
            end
        elseif picks[j,6] == "S" && picks[j,5] > 0.6
            if uobs_s[sta_idx,i] == -1 || abs(delta_time-scaltime_s[sta_idx,i]) < abs(uobs_s[sta_idx,i] - scaltime_s[sta_idx,i])
                uobs_s[sta_idx,i] = delta_time
                qua_s[sta_idx,i] = picks[j,5]
            end
        end
    end
end

delt_p = []; delt_s = []; numbig_p = 0; numsmall_p = 0; numbig_s = 0; numsmall_s = 0
sta_record_p = zeros(numsta,2); eve_record_p = zeros(numeve,2)
sta_record_s = zeros(numsta,2); eve_record_s = zeros(numeve,2)
for i = 1:numeve
    for j = 1:numsta
        if uobs_p[j,i] != -1
            push!(delt_p,uobs_p[j,i]-scaltime_p[j,i])
            if abs(uobs_p[j,i] - scaltime_p[j,i]) < prange 
                if (uobs_p[j,i]-scaltime_p[j,i])>0
                    global numbig_p += 1
                    sta_record_p[j,1] += 1; eve_record_p[i,1] += 1
                end
                if (uobs_p[j,i]-scaltime_p[j,i])<0
                    global numsmall_p += 1
                    sta_record_p[j,2] += 1; eve_record_p[i,2] += 1
                end
            else 
                uobs_p[j,i] = -1
            end
        end
        if uobs_s[j,i] != -1 
            push!(delt_s,uobs_s[j,i]-scaltime_s[j,i])
            if abs(uobs_s[j,i] - scaltime_s[j,i]) < srange 
                if (uobs_s[j,i]-scaltime_s[j,i])>0
                    global numbig_s += 1
                    sta_record_s[j,1] += 1; eve_record_s[i,1] += 1
                end
                if (uobs_s[j,i]-scaltime_s[j,i])<0
                    global numsmall_s += 1
                    sta_record_s[j,2] += 1; eve_record_s[i,2] += 1
                end
            else 
                uobs_s[j,i] = -1
            end
        end
    end
end
print(numbig_p," ",numsmall_p,'\n',numbig_s," ",numsmall_s,'\n')
sum_p = 0
for delt in delt_p
    if abs(delt) < prange
        global sum_p += delt^2
    end
end
sum_s = 0
for delt in delt_s
    if abs(delt) < srange
        global sum_s += delt^2
    end
end
print(sum_p," ",sum_s,'\n')
sta_ratio_p = ones(numsta); eve_ratio_p = ones(numeve)
sta_ratio_s = ones(numsta); eve_ratio_s = ones(numeve)
for i = 1:numsta
    sta_ratio_p[i] = (sta_record_p[i,1]-sta_record_p[i,2]) / (sta_record_p[i,1]+sta_record_p[i,2])
    sta_ratio_s[i] = (sta_record_s[i,1]-sta_record_s[i,2]) / (sta_record_s[i,1]+sta_record_s[i,2])
end
for i = 1:numeve
    eve_ratio_p[i] = (eve_record_p[i,1]-eve_record_p[i,2]) / (eve_record_p[i,1]+eve_record_p[i,2])
    eve_ratio_s[i] = (eve_record_s[i,1]-eve_record_s[i,2]) / (eve_record_s[i,1]+eve_record_s[i,2])
end

plt.figure(); plt.hist(delt_p,bins=5000,edgecolor="royalblue",color="skyblue")
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-prange,prange)
plt.title("Histogram_P");plt.savefig(folder * "for_P/hist_p_0.png")
plt.figure(); plt.hist(delt_s,bins=8000,edgecolor="royalblue",color="skyblue"); 
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-srange,srange)
plt.title("Histogram_S");plt.savefig(folder * "for_S/hist_s_0.png")

cover_p = zeros(m,n,l); cover_s=zeros(m,n,l)
for i = 1:numeve
    for j = 1:numsta
        if uobs_s[j,i] == -1 && uobs_p[j,i] == -1
            continue
        end
        length = max(abs(alleve.x[i]-allsta.x[j]),abs(alleve.y[i]-allsta.y[j]),abs(alleve.z[i]-allsta.z[j]))
        local dx = (alleve.x[i]-allsta.x[j])/length
        local dy = (alleve.y[i]-allsta.y[j])/length
        local dz = (alleve.z[i]-allsta.z[j])/length
        len = 0; nx = allsta.x[j]; ny = allsta.y[j]; nz = allsta.z[j]
        while len<length && nx>1 && nx<m && ny>1 && ny<n && nz>1 && nz<l
            if uobs_p[j,i] != -1
                cover_p[convert(Int,floor(nx)),convert(Int,floor(ny)),convert(Int,floor(nz))] += 1
            end
            if uobs_s[j,i] != -1
                cover_s[convert(Int,floor(nx)),convert(Int,floor(ny)),convert(Int,floor(nz))] += 1
            end
            nx += dx; ny += dy; nz += dz; len += 1
        end 
    end
end
for i = 1:m
    for j = 1:n
        for k = 1:l
            cover_p[i,j,k] = log10(cover_p[i,j,k])
            cover_s[i,j,k] = log10(cover_s[i,j,k])
        end
    end
end

figure(figsize=(20,20))
for i = 1:min(25,l)
    subplot(5,5,i)
    pcolormesh(cover_p[:,:,i],cmap="afmhot_r")
    colorbar()
end
savefig(folder * "for_P/coverage_p.png")
figure(figsize=(20,20))
for i = 1:min(25,l)
    subplot(5,5,i)
    pcolormesh(cover_s[:,:,i],cmap="afmhot_r")
    colorbar()
end
savefig(folder * "for_S/coverage_s.png")
for i = 1:min(25,l)
    figure(figsize=(5,5))
    pcolormesh(cover_p[:,:,i],cmap="afmhot_r")
    colorbar()
    savefig(folder * "for_P/coverage/layer_$i.png")
    close()
    figure(figsize=(5,5))
    pcolormesh(cover_s[:,:,i],cmap="afmhot_r")
    colorbar()
    savefig(folder * "for_S/coverage/layer_$i.png")
    close()
end

h5write(folder * "for_P/uobs_p.h5","matrix",uobs_p)
h5write(folder * "for_S/uobs_s.h5","matrix",uobs_s)
h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
h5write(folder * "for_S/qua_s.h5","matrix",qua_s)
h5write(folder * "for_P/eve_ratio_p.h5","data",eve_ratio_p)
h5write(folder * "for_P/sta_ratio_p.h5","data",sta_ratio_p)
h5write(folder * "for_S/eve_ratio_s.h5","data",eve_ratio_s)
h5write(folder * "for_S/sta_ratio_s.h5","data",sta_ratio_s)