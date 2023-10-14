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

region = "demo/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile(folder * "config.json")["gene_obs"]
prange = config["prange"]; srange=config["srange"]
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

stations = CSV.read("../local/"*region*"seismic_data/obspy/stations.csv", DataFrame)
events = CSV.read("../local/"*region*"seismic_data/obspy/catalog.csv", DataFrame)
allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)
file = open(folder * "sta_eve/stations.json", "r"); dic_sta = JSON.parse(file); close(file)
eveid = h5read(folder * "sta_eve/eveid.h5","data")

vel_p = h5read(folder * "velocity/GIL7_vel0_p.h5","data") .* config["p_times"]
vel_s = h5read(folder * "velocity/GIL7_vel0_s.h5","data") .* config["s_times"]
if isfile(folder * "velocity/vel0_p.h5")
    rm(folder * "velocity/vel0_p.h5")
    rm(folder * "velocity/vel0_s.h5")
end
h5write(folder * "velocity/vel0_p.h5","data",vel_p)
h5write(folder * "velocity/vel0_s.h5","data",vel_s)

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
    #local file = "../local/"*region*"seismic_data/phasenet/picks/" * id * ".csv"
    local file = "../local/"*region*"seismic_data/phasenet/picks_phasenet/" * id * ".csv"
    local picks = CSV.read(file,DataFrame)
    local numpick = size(picks,1)

    for j = 1:numpick
        local sta_idx = dic_sta[picks[j,1]]
        if sta_idx == -1  
            continue   
        end

        local delta_militime = picks[j,3] - parse(DateTime,evetime[1:23])
        local delta_time = delta_militime.value/1000
        if picks[j,5] == "P" && picks[j,4] > config["p_requirement"]
            if uobs_p[sta_idx,i] == -1 || abs(delta_time-scaltime_p[sta_idx,i]) < abs(uobs_p[sta_idx,i] - scaltime_p[sta_idx,i])
                uobs_p[sta_idx,i] = delta_time
                qua_p[sta_idx,i] = picks[j,4]
            end
        elseif picks[j,5] == "S" && picks[j,4] > config["s_requirement"]
            if uobs_s[sta_idx,i] == -1 || abs(delta_time-scaltime_s[sta_idx,i]) < abs(uobs_s[sta_idx,i] - scaltime_s[sta_idx,i])
                uobs_s[sta_idx,i] = delta_time
                qua_s[sta_idx,i] = picks[j,4]
            end
        end
    end
end

if !isdir(folder*"/for_P/") 
    mkdir(folder*"/for_P/")
    mkdir(folder*"/for_S/")
end

if isfile(folder * "for_P/uobs_p.h5")
    rm(folder * "for_P/uobs_p.h5")
    rm(folder * "for_P/qua_p.h5")
    rm(folder * "for_S/uobs_s.h5")
    rm(folder * "for_S/qua_s.h5")
end
h5write(folder * "for_P/uobs_p.h5","matrix",uobs_p)
h5write(folder * "for_S/uobs_s.h5","matrix",uobs_s)
h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
h5write(folder * "for_S/qua_s.h5","matrix",qua_s)

delt_p = []; delt_s = []; numbig_p = 0; numsmall_p = 0; numbig_s = 0; numsmall_s = 0
sta_record_p = zeros(numsta,2); eve_record_p = zeros(numeve,2);sum_p = 0
sta_record_s = zeros(numsta,2); eve_record_s = zeros(numeve,2);sum_s = 0
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
                global sum_p += qua_p[j,i] * (uobs_p[j,i]-scaltime_p[j,i])^2
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
                global sum_s += qua_s[j,i] * (uobs_s[j,i]-scaltime_s[j,i])^2
            else 
                uobs_s[j,i] = -1
            end
        end
    end
end

print(numbig_p," ",numsmall_p,'\n',numbig_s," ",numsmall_s,'\n')
print(sum_p," ",sum_s,'\n')
plt.figure(); plt.hist(delt_p,bins=config["bins_p"],edgecolor="royalblue",color="skyblue");
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-prange,prange)
plt.savefig(folder * "for_P/hist_p_0.png")
plt.figure(); plt.hist(delt_s,bins=config["bins_s"],edgecolor="royalblue",color="skyblue"); 
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-srange,srange)
plt.savefig(folder * "for_S/hist_s_0.png")
#

sta_median_p = zeros(numsta); sta_median_s = zeros(numsta)
for i = 1:numsta
    local sta_delt_p = []; local sta_delt_s = []
    for j = 1:numeve
        if uobs_p[i,j] != -1
            push!(sta_delt_p, uobs_p[i,j]-scaltime_p[i,j])
        end
        if uobs_s[i,j] != -1
            push!(sta_delt_s, uobs_s[i,j]-scaltime_s[i,j])
        end
    end
    local num = size(sta_delt_p,1)
    if num == 0 continue
    elseif num % 2 == 0 sta_median_p[i] = sort(sta_delt_p)[convert(Int,num/2)]
    else sta_median_p[i] = sort(sta_delt_p)[convert(Int,(num+1)/2)]
    end
    local num = size(sta_delt_s,1)
    if num == 0 continue
    elseif num % 2 == 0 sta_median_s[i] = sort(sta_delt_s)[convert(Int,num/2)]
    else sta_median_s[i] = sort(sta_delt_s)[convert(Int,(num+1)/2)]
    end
end
eve_median_p = zeros(numeve); eve_median_s = zeros(numeve)
for i = 1:numeve
    local eve_delt_p = []; local eve_delt_s = []
    for j = 1:numsta
        if uobs_p[j,i] != -1
            push!(eve_delt_p, uobs_p[j,i]-scaltime_p[j,i])
        end
        if uobs_s[j,i] != -1
            push!(eve_delt_s, uobs_s[j,i]-scaltime_s[j,i])
        end
    end
    local num = size(eve_delt_p,1)
    if num == 0 continue
    elseif num % 2 == 0 eve_median_p[i] = sort(eve_delt_p)[convert(Int,num/2)]
    else eve_median_p[i] = sort(eve_delt_p)[convert(Int,(num+1)/2)]
    end
    local num = size(eve_delt_s,1)
    if num == 0 continue
    elseif num % 2 == 0 eve_median_s[i] = sort(eve_delt_s)[convert(Int,num/2)]
    else eve_median_s[i] = sort(eve_delt_s)[convert(Int,(num+1)/2)]
    end
end

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
if !isdir(folder * "for_P/residual")
    mkdir(folder * "for_P/residual")
    mkdir(folder * "for_S/residual")
end
file = open(folder * "for_P/residual/eve_ratio_p.txt","w")
for i = 1:numeve
    println(file, eve_ratio_p[i])
end
close(file)
file = open(folder * "for_P/residual/sta_ratio_p.txt","w")
for i = 1:numsta
    println(file, sta_ratio_p[i])
end
close(file)
file = open(folder * "for_P/residual/eve_median_p.txt","w")
for i = 1:numeve
    println(file, eve_median_p[i])
end
close(file)
file = open(folder * "for_P/residual/sta_median_p.txt","w")
for i = 1:numsta
    println(file, sta_median_p[i])
end
close(file)
file = open(folder * "for_S/residual/eve_ratio_s.txt","w")
for i = 1:numeve
    println(file, eve_ratio_s[i])
end
close(file)
file = open(folder * "for_S/residual/sta_ratio_s.txt","w")
for i = 1:numsta
    println(file, sta_ratio_s[i])
end
close(file)
file = open(folder * "for_S/residual/eve_median_s.txt","w")
for i = 1:numeve
    println(file, eve_median_s[i])
end
close(file)
file = open(folder * "for_S/residual/sta_median_s.txt","w")
for i = 1:numsta
    println(file, sta_median_s[i])
end
close(file)
#

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
if !isdir(folder * "for_P/coverage/")
    mkdir(folder * "for_P/coverage/")
    mkdir(folder * "for_S/coverage/")
end
for i = 1:min(25,l)
    figure(figsize=(4,8))
    pcolormesh(transpose(cover_p[:,:,i]),cmap="magma_r")
    colorbar()
    savefig(folder * "for_P/coverage/layer_$i.png")
    close()
    figure(figsize=(4,8))
    pcolormesh(transpose(cover_s[:,:,i]),cmap="magma_r")
    colorbar()
    savefig(folder * "for_S/coverage/layer_$i.png")
    close()
end
