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

region = "demo/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile(folder * "config.json")["inversion"]
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)
uobs = h5read(folder * "for_P/uobs_p.h5","matrix")
qua = h5read(folder * "for_P/qua_p.h5","matrix")

vel0 = h5read(folder * "inv_P_0.005/post/post_100.h5","data")
uvar = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))

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

delt = []; sum_res = 0; numbig = 0; numsmall = 0
sta_record = zeros(numsta,2); sta_median = zeros(numsta); sta_ratio = ones(numsta)
for i = 1:numsta
    local sta_delt = []
    for j = 1:numeve
        if uobs[i,j] != -1
            push!(sta_delt, uobs[i,j]-caltime[i,j])
            push!(delt,uobs[i,j]-caltime[i,j])
            global sum_res += qua[i,j] * (uobs[i,j]-caltime[i,j])^2
            if (uobs[i,j]-caltime[i,j]) > 0
                global numbig += 1
                sta_record[i,1] += 1
            end
            if (uobs[i,j]-caltime[i,j]) < 0
                global numsmall += 1
                sta_record[i,2] += 1
            end 
        end
    end
    local sta_num = size(sta_delt,1)
    if sta_num == 0 continue
    elseif sta_num%2 == 0 sta_median[i] = sort(sta_delt)[convert(Int,sta_num/2)]
    else sta_median[i] = sort(sta_delt)[convert(Int,(sta_num+1)/2)] end
end
for i = 1:numsta
    sta_ratio[i] = (sta_record[i,1]-sta_record[i,2]) / (sta_record[i,1]+sta_record[i,2])
end
print(numbig," ",numsmall,'\n')

folder = folder * "inv_P_" * string(config["lambda_p"]) * "/final/"
if !isdir(folder) mkdir(folder) end

file = open(folder * "sta_ratio_s.txt","w")
for i = 1:numsta
    println(file, sta_ratio[i])
end
close(file)
file = open(folder * "sta_median_s.txt","w")
for i = 1:numsta
    println(file, sta_median[i])
end
close(file)

plt.figure(); plt.hist(delt,bins=60,edgecolor="royalblue",color="skyblue");
plt.xlabel("Residual"); plt.ylabel("Counts")
plt.xlim(-3,3); plt.ylim(0,3000)
plt.savefig(folder * "histogram_p.png")
@show sum_res