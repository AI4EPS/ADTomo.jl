push!(LOAD_PATH,"../src")
using CSV
using DataFrames
using Serialization
using HDF5
using ADCME
using ADTomo
using PyCall
using Dates
using PyPlot

folder = "readin_data/no_rotation/"
h = 1.0; bins = 2000; prange = 2; srange = 5; theta = 32
veltimes_p = 0.99; veltimes_s = 1.04; phasereq = 0.8

raw_stations = CSV.read("seismic_data/BayArea/obspy/stations.csv", DataFrame)
events = CSV.read("seismic_data/BayArea/obspy/catalog.csv", DataFrame)
numsta = size(raw_stations,1); numeve = size(events,1)
stations = DataFrame(x=[], y=[], z=[], lon=[], lat=[])
dic_sta = Dict(); numsta_ = 0
for j = 1:numsta
    local flag = false
    if raw_stations[j,3]===missing
        local name = raw_stations[j,1]*'.'*raw_stations[j,2]*".."*raw_stations[j,5]
    else
        local name = raw_stations[j,1]*'.'*raw_stations[j,2]*'.'*raw_stations[j,3]*'.'*raw_stations[j,5]
    end
    if haskey(dic_sta,name) 
        continue 
    end
    for k = numsta_:-1:1
        if raw_stations[j,18]==stations.x[k] && raw_stations[j,19]==stations.y[k] && raw_stations[j,20]==stations.z[k]
            dic_sta[name] = k
            flag = true
            continue
        end
    end
    if flag continue end
    global numsta_ += 1
    dic_sta[name] = numsta_
    push!(stations.x, raw_stations[j,18])
    push!(stations.y, raw_stations[j,19])
    push!(stations.z, raw_stations[j,20])
    push!(stations.lon,raw_stations[j,7])
    push!(stations.lat,raw_stations[j,8])
end
numsta = numsta_

# study a region of (m=31km, n=41km, l=30km(-5~25))
minx = 0; miny = 0; minz = 0;
maxx = 0; maxy = 0; maxz = 0;

allsta = DataFrame(x = [], y = [], z = [], lon = [], lat = [])
alleve = DataFrame(x = [], y = [], z = [], lon = [], lat = [])
staget = zeros(numsta);eveid = []
for i = 1:numeve
    file = "seismic_data/BayArea/phasenet/picks/" * events[i,1] * ".csv"
    if !isfile(file) 
        continue 
    end

    local eveget = 0
    local stajudge = zeros(numsta)
    picks = CSV.read(file,DataFrame); numpick = size(picks,1)
    for j = 1:numpick
        if picks[j,5] < phasereq
            continue
        end
        sta_id = dic_sta[picks[j,1]]
        if stajudge[sta_id] == 1
            continue
        end
        stajudge[sta_id] = 1
        staget[sta_id] += 1
        eveget += 1
    end
    if eveget < 10 
        continue
    end
    x = (events[i,7] * cosd(theta) + events[i,8] * sind(theta))/h
    y = (events[i,8] * cosd(theta) - events[i,7] * sind(theta))/h
    z = events[i,9]/h
    if x<-60 || x>100 || z>25 continue end

    push!(eveid,i)
    push!(alleve.x,x); push!(alleve.y,y); push!(alleve.z,z) 
    push!(alleve.lon,events[i,5]); push!(alleve.lat,events[i,4])

    global minx = min(minx,x); global miny = min(miny,y); global minz = min(minz,z)
    global maxx = max(maxx,x); global maxy = max(maxy,y); global maxz = max(maxz,z)
end
numeve = size(alleve,1)

maxstaget = maximum(staget); numsta_ = 0; dic_new = Dict()
for j = 1:numsta
    if staget[j] < 5
        dic_new[j] = -1
        continue
    end
    x = (stations.x[j]*cosd(theta) + stations.y[j]*sind(theta))/h
    y = (stations.y[j]*cosd(theta) - stations.x[j]*sind(theta))/h
    z = stations.z[j]/h
    if x<-60 || x>100 
        dic_new[j] = -1
        continue  
    end

    global numsta_ += 1; dic_new[j] = numsta_
    push!(allsta.x,x); push!(allsta.y,y); push!(allsta.z,z)
    push!(allsta.lon,stations.lon[j]); push!(allsta.lat,stations.lat[j])

    global minx = min(minx,x); global miny = min(miny,y); global minz = min(minz,z)
    global maxx = max(maxx,x); global maxy = max(maxy,y); global maxz = max(maxz,z)
end

dx = convert(Int64, ceil(abs(minx)) + h); m = convert(Int64,ceil(maxx + dx) + h)
dy = convert(Int64, ceil(abs(miny)) + h); n = convert(Int64,ceil(maxy + dy) + h)
dz = convert(Int64, ceil(abs(minz)) + h); l = convert(Int64,ceil(maxz + dz) + h)
for i = 1:numsta_
    allsta.x[i] += dx; allsta.y[i] += dy; allsta.z[i] += dz
end
for i = 1:numeve
    alleve.x[i] += dx; alleve.y[i] += dy; alleve.z[i] += dz
end

vel_p = Dict(); fvel0_p = ones(m,n,l); vel0_p = ones(m,n,l) 
vel_s = Dict(); fvel0_s = ones(m,n,l); vel0_s = ones(m,n,l)

vel_p[0] = 3.20; vel_p[1] = 4.50; vel_p[3] = 4.80; vel_p[4] = 5.51
vel_p[5] = 6.21; vel_p[17] = 6.89; vel_p[25] = 7.83
nvel = vel_p[0] * veltimes_p
for i = 1:l
    if i*h-dz in keys(vel_p)
        global nvel = vel_p[i*h-dz] * veltimes_p
    end
    fvel0_p[:,:,i] .= 1/nvel
    vel0_p[:,:,i] .= nvel
end
vel_s[0] = 1.50; vel_s[1] = 2.40; vel_s[3] = 2.78; vel_s[4] = 3.18
vel_s[5] = 3.40; vel_s[17] = 3.98; vel_s[25] = 4.52
nvel = vel_s[0] * veltimes_s
for i = 1:l
    if i*h-dz in keys(vel_s)
        global nvel = vel_s[i*h-dz] * veltimes_s
    end
    fvel0_s[:,:,i] .= 1/nvel
    vel0_s[:,:,i] .= nvel
end

u_p = PyObject[]; u_s = PyObject[]
for i = 1:numsta_
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    slowness_pu = fvel0_p[ixu,iyu,izu]; slowness_pd = fvel0_p[ixu,iyu,izd]
    slowness_su = fvel0_s[ixu,iyu,izu]; slowness_sd = fvel0_s[ixu,iyu,izd]
    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    push!(u_p,eikonal3d(u0,fvel0_p,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    push!(u_s,eikonal3d(u0,fvel0_s,h,m,n,l,1e-3,false))
end
sess = Session(); init(sess)
ubeg_p = run(sess,u_p); ubeg_s = run(sess,u_s)

scaltime_p = ones(numsta_,numeve); scaltime_s = ones(numsta_,numeve)
for i = 1:numsta_
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

## read observed time
uobs_p = -ones(numsta_,numeve); uobs_s = -ones(numsta_,numeve)
qua_p = ones(numsta_,numeve); qua_s = ones(numsta_,numeve)

for i = 1:numeve
    local evetime = events[eveid[i],2]; local id = events[eveid[i],1]
    local file = "seismic_data/BayArea/phasenet/picks/" * id * ".csv"
    picks = CSV.read(file,DataFrame)
    numpick = size(picks,1)

    for j = 1:numpick
        sta_idx = dic_new[dic_sta[picks[j,1]]]
        if sta_idx == -1  
            continue   
        end
        if picks[j,5] < phasereq 
            continue 
        end

        delta_militime = picks[j,4] - parse(DateTime,evetime[1:23])
        delta_time = delta_militime.value/1000
        if picks[j,6] == "P" 
            if uobs_p[sta_idx,i] == -1 || abs(delta_time-scaltime_p[sta_idx,i]) < abs(uobs_p[sta_idx,i] - scaltime_p[sta_idx,i])
                uobs_p[sta_idx,i] = delta_time
                qua_p[sta_idx,i] = picks[j,5]
            end
        elseif picks[j,6] == "S"
            if uobs_s[sta_idx,i] == -1 || abs(delta_time-scaltime_s[sta_idx,i]) < abs(uobs_s[sta_idx,i] - scaltime_s[sta_idx,i])
                uobs_s[sta_idx,i] = delta_time
                qua_s[sta_idx,i] = picks[j,5]
            end
        end
    end
end

## plot residual histogram
delt_p = []; delt_s = []; numbig_p = 0; numsmall_p = 0; numbig_s = 0; numsmall_s = 0
sta_record_p = zeros(numsta_,2); eve_record_p = zeros(numeve,2)
sta_record_s = zeros(numsta_,2); eve_record_s = zeros(numeve,2)
sta_sum = zeros(numsta_); eve_sum = zeros(numeve)
for i = 1:numeve
    for j = 1:numsta_
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
                sta_sum[j] += uobs_p[j,i] - scaltime_p[j,i]
                eve_sum[i] += uobs_p[j,i] - scaltime_p[j,i]
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
                sta_sum[j] += uobs_s[j,i] - scaltime_s[j,i]
                eve_sum[i] += uobs_s[j,i] - scaltime_s[j,i]
            else 
                uobs_s[j,i] = -1
            end
        end
    end
end
sta_ratio_p = ones(numsta_); eve_ratio_p = ones(numeve)
sta_ratio_s = ones(numsta_); eve_ratio_s = ones(numeve)
for i = 1:numsta_
    sta_ratio_p[i] = (sta_record_p[i,1]-sta_record_p[i,2]) / (sta_record_p[i,1]+sta_record_p[i,2])
    sta_ratio_s[i] = (sta_record_s[i,1]-sta_record_s[i,2]) / (sta_record_s[i,1]+sta_record_s[i,2])
end
for i = 1:numeve
    eve_ratio_p[i] = (eve_record_p[i,1]-eve_record_p[i,2]) / (eve_record_p[i,1]+eve_record_p[i,2])
    eve_ratio_s[i] = (eve_record_s[i,1]-eve_record_s[i,2]) / (eve_record_s[i,1]+eve_record_s[i,2])
end
sta_sum /= maximum(sta_sum); eve_sum /= maximum(eve_sum)

print(numbig_p,' ',numsmall_p,'\n',numbig_s,' ',numsmall_s,'\n')
print(numsta,' ',numeve,'\n',size(stations,1),' ',size(events,1),'\n')
#
rfile = open(folder * "/range.txt","w")
println(rfile,m);println(rfile,n);println(rfile,l);
println(rfile,h);println(rfile,bins)
println(rfile,dx);println(rfile,dy);println(rfile,dz)
close(rfile)
#= for save matrixes

CSV.write(folder * "alleve.csv",alleve)
CSV.write(folder * "allsta.csv",allsta)
h5write(folder * "eve_ratio_p.h5","data",eve_ratio_p)
h5write(folder * "sta_ratio_p.h5","data",sta_ratio_p)
h5write(folder * "eve_ratio_s.h5","data",eve_ratio_s)
h5write(folder * "sta_ratio_s.h5","data",sta_ratio_s)
h5write(folder * "eve_sum.h5","data",eve_sum)
h5write(folder * "sta_sum.h5","data",sta_sum)

h5write(folder * "1D_fvel0_p.h5","data",fvel0_p)
h5write(folder * "1D_fvel0_s.h5","data",fvel0_s)
h5write(folder * "1D_vel0_p.h5","data",vel0_p)
h5write(folder * "1D_vel0_s.h5","data",vel0_s)
h5write(folder * "for_P/scaltime_p.h5","matrix",scaltime_p)
h5write(folder * "for_S/scaltime_s.h5","matrix",scaltime_s)
h5write(folder * "for_P/uobs_p.h5","matrix",uobs_p)
h5write(folder * "for_S/uobs_s.h5","matrix",uobs_s)
h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
h5write(folder * "for_S/qua_s.h5","matrix",qua_s)
=#
figure()
scatter(alleve.y,alleve.x)
scatter(allsta.y,allsta.x)
savefig(folder * "/location.png")

plt.figure(); plt.hist(delt_p,bins=bins)
plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.xlim(-prange*2,prange*2)
plt.title("Histogram_P");plt.savefig(folder * "for_P/hist_p.png")
plt.figure(); plt.hist(delt_s,bins=bins); plt.xlim(-srange*2,srange*2)
plt.xlabel("Residual"); plt.ylabel("Frequency")
plt.title("Histogram_S");plt.savefig(folder * "for_S/hist_s.png")