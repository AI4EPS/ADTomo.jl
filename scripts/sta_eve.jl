using CSV
using DataFrames
using HDF5
using PyCall
using PyPlot
using Clustering
using Distances
using JSON

region = "demo/"
config = JSON.parsefile("../local/" * region * "readin_data/config.json")["sta_eve"]
h = config["h"]; theta = config["theta"]
folder = "../local/" * region * "seismic_data/"
events = CSV.read(folder * "obspy/catalog.csv", DataFrame); numeve = size(events,1)
raw_stations = CSV.read(folder * "obspy/stations.csv", DataFrame); numsta = size(raw_stations,1)
stations = DataFrame(x=[], y=[], z=[], lon=[], lat=[]); dic_sta = Dict(); numsta_ = 0
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

minx = 0; miny = 0; minz = 0; maxx = 0; maxy = 0; maxz = 0
allsta = DataFrame(x = [], y = [], z = [], lon = [], lat = [])
alleve = DataFrame(x = [], y = [], z = [], lon = [], lat = [], picks = [])
staget = zeros(numsta); eveid = Vector{Int}()
for i = 1:numeve
    local file1 = folder * "phasenet/picks_phasenet/" * events[i,1] * ".csv"
    if !isfile(file1) 
        continue 
    end

    local eveget = 0; local stajudge = zeros(numsta)
    local picks = CSV.read(file1,DataFrame); local numpick = size(picks,1)
    for j = 1:numpick
        if picks[j,4] < config["p_requirement"] && picks[j,5] == "P"
            continue
        end
        if picks[j,4] < config["s_requirement"] && picks[j,5] == "S"
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
    if eveget < config["eve_picks"]
        continue
    end
    x = (events[i,7] * cosd(theta) + events[i,8] * sind(theta))/h
    y = (events[i,8] * cosd(theta) - events[i,7] * sind(theta))/h
    z = events[i,9]/h
    if x<config["s_x"] || x>config["l_x"] || y<config["s_y"] || y>config["l_y"] || z<config["s_z"] || z>config["l_z"] 
        continue
    end

    push!(eveid, i); push!(alleve.picks, eveget)
    push!(alleve.x, x); push!(alleve.y,y); push!(alleve.z, z) 
    push!(alleve.lon, events[i,5]); push!(alleve.lat, events[i,4])

    global minx = min(minx,x); global miny = min(miny,y); global minz = min(minz,z)
    global maxx = max(maxx,x); global maxy = max(maxy,y); global maxz = max(maxz,z)
end
numeve = size(alleve,1)

numsta_ = 0; dic_new = Dict()
for j = 1:numsta
    if staget[j] < config["sta_picks"]
        dic_new[j] = -1
        continue
    end
    x = (stations.x[j]*cosd(theta) + stations.y[j]*sind(theta))/h
    y = (stations.y[j]*cosd(theta) - stations.x[j]*sind(theta))/h
    z = stations.z[j]/h
    if x<config["s_x"] || x>config["l_x"] || y<config["s_y"] || y>config["l_y"]
        dic_new[j] = -1
        continue  
    end

    global numsta_ += 1; dic_new[j] = numsta_
    push!(allsta.x,x); push!(allsta.y,y); push!(allsta.z,z)
    push!(allsta.lon,stations.lon[j]); push!(allsta.lat,stations.lat[j])

    global minx = min(minx,x); global miny = min(miny,y); global minz = min(minz,z)
    global maxx = max(maxx,x); global maxy = max(maxy,y); global maxz = max(maxz,z)
end
dic_new[-1] = -1
dic_filter = Dict()
for key in keys(dic_sta)
    dic_filter[key] = dic_new[dic_sta[key]]
end

dx = convert(Int64, ceil(abs(minx)) + 1); m = convert(Int64,ceil(maxx + dx) + 1)
dy = convert(Int64, ceil(abs(miny)) + 1); n = convert(Int64,ceil(maxy + dy) + 1)
dz = convert(Int64, ceil(abs(minz)) + 1); l = convert(Int64,ceil(maxz + dz) + 1)
for i = 1:numsta_
    allsta.x[i] += dx; allsta.y[i] += dy; allsta.z[i] += dz
end
for i = 1:numeve
    alleve.x[i] += dx; alleve.y[i] += dy; alleve.z[i] += dz
end

folder = "../local/" * region * "readin_data/"
if !isdir(folder) mkdir(folder) end 
rfile = open(folder * "range.txt","w")
println(rfile,m);println(rfile,n);println(rfile,l);println(rfile,h)
println(rfile,dx);println(rfile,dy);println(rfile,dz)
close(rfile)

eve_data = Vector{Vector{Float64}}()
for i = 1:numeve
    push!(eve_data,[alleve.x[i],alleve.y[i],alleve.z[i]])
end
dist_matrix = pairwise(Euclidean(), eve_data)
eps = config["eve_eps"]; min_pts = 1
result = dbscan(dist_matrix, eps, min_pts)
clusters = assignments(result)
cluster_points = Dict{Int, Vector{Int}}()

for (idx, cluster) in enumerate(clusters)
    if cluster != -1
        if haskey(cluster_points, cluster)
            push!(cluster_points[cluster], idx)
        else
            cluster_points[cluster] = [idx]
        end
    end
end

newid = Vector{Int}()
for (cluster,points) in cluster_points
    if size(points,1) == 1
        push!(newid,points[1])
    else
        local picks = []
        for point in points
            push!(picks,alleve.picks[point])
        end
        num = convert(Int,ceil(size(picks,1)/config["eve_ratio"]))
        sort_picks = sortperm(picks,rev=true)
        for i = 1:num
            push!(newid,points[sort_picks[i]])
        end
    end
end
numeve_ = size(newid,1)

eve_new = DataFrame(x=[], y=[], z=[], lon=[], lat=[])
eveid_new = Vector{Int}()
for i = 1:numeve_
    push!(eve_new.x, alleve.x[newid[i]])
    push!(eve_new.y, alleve.y[newid[i]])
    push!(eve_new.z, alleve.z[newid[i]])
    push!(eve_new.lon, alleve.lon[newid[i]])
    push!(eve_new.lat, alleve.lat[newid[i]])
    push!(eveid_new, eveid[newid[i]])
end

sta_data = Vector{Vector{Float64}}()
for i = 1:numsta_
    push!(sta_data,[allsta.x[i],allsta.y[i],allsta.z[i]])
end
dist_matrix = pairwise(Euclidean(), sta_data)
eps = 1.414; min_pts = 1
result = dbscan(dist_matrix, eps, min_pts)
clusters = assignments(result)
cluster_points = Dict{Int, Vector{Int}}()
for (idx, cluster) in enumerate(clusters)
    if cluster != -1
        if haskey(cluster_points, cluster)
            push!(cluster_points[cluster], idx)
        else
            cluster_points[cluster] = [idx]
        end
    end
end
numsta = length(keys(cluster_points))

dic_inv = Dict(); dic_record = Dict()
for key in keys(dic_new)
    num = dic_new[key]
    if num == -1 continue end
    dic_inv[num] = key
end
sta_new = DataFrame(x=[], y=[], z=[], lon=[], lat=[])
for i = 1:numsta
    points = cluster_points[i]
    maxpick = 0; record = 0
    if size(points,1) == 1 
        record = points[1]
    else
        for point in points
            if staget[dic_inv[point]] > maxpick
                maxpick = staget[dic_inv[point]]
                record = point
            end
        end
    end
    push!(sta_new.x, allsta.x[record])
    push!(sta_new.y, allsta.y[record])
    push!(sta_new.z, allsta.z[record])
    push!(sta_new.lon, allsta.lon[record])
    push!(sta_new.lat, allsta.lat[record])
    dic_record[record] = i
    for point in points
        if point != record
            dic_record[point] = -1
        end
    end
end
dic_record[-1] = -1
dic_save = Dict()
for key in keys(dic_filter)
    dic_save[key] = dic_record[dic_filter[key]]
end

folder = folder * "sta_eve/"
if !isdir(folder) mkdir(folder) end 
CSV.write(folder * "alleve.csv",eve_new)
CSV.write(folder * "allsta.csv",sta_new)
file2 = open(folder * "stations.json", "w")
JSON.print(file2, dic_save); close(file2)
h5write(folder * "eveid.h5", "data", eveid_new)

figure()
scatter(eve_new.y,eve_new.x,label="events"); plt.legend()
scatter(sta_new.y,sta_new.x,label="stations");plt.legend()
savefig(folder * "location.png")
#