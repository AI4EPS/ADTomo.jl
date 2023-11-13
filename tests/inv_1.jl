push!(LOAD_PATH,"/home/lingxia/ADTomo.jl/src/")
using ADCME
using ADTomo
using PyCall
using PyPlot
using LinearAlgebra
using Serialization
using DataFrames
using CSV
using HDF5
using Random
using Optim
Random.seed!(233)

reset_default_graph()

region = "demo/"
folder = "../local/" * region * "readin_data/"
prange = 2; srange = 5
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
rpvs = 1.2; bins = 500

allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)

fvel0_p = h5read(folder * "velocity/vel0_p.h5","data")
#fvel0_s = h5read(folder * "for_S/1D_fvel0_s.h5","data")
uobs_p = h5read(folder * "/for_P/uobs_p.h5","matrix")
uobs_s = h5read(folder * "/for_S/uobs_s.h5","matrix")
qua_p = h5read(folder * "/for_P/qua_p.h5","matrix")
qua_s = h5read(folder * "/for_S/qua_s.h5","matrix")

fvar = fvel0_p + 2*sigmoid(Variable(zero(fvel0_p)))-1; pvs = Variable(rpvs)
sess = Session(); init(sess)
uvar_p = PyObject[]; uvar_s = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel0_p[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel0_p[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel0_p[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel0_p[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel0_p[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel0_p[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel0_p[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel0_p[ixd,iyd,izd]
    push!(uvar_p,eikonal3d(u0,1 ./ fvar,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel0_p[ixu,iyu,izu]*rpvs
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel0_p[ixu,iyu,izd]*rpvs
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel0_p[ixu,iyd,izu]*rpvs
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel0_p[ixu,iyd,izd]*rpvs
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel0_p[ixd,iyu,izu]*rpvs
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel0_p[ixd,iyu,izd]*rpvs
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel0_p[ixd,iyd,izu]*rpvs
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel0_p[ixd,iyd,izd]*rpvs
    push!(uvar_s,eikonal3d(u0,pvs ./ fvar,h,m,n,l,1e-3,false))
end

caltime_p = []; caltime_s = []
for i = 1:numsta
    timei_p = []; timei_s = []
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

        if x1 == x2
            tx11 = uvar_s[i][x1,y1,z1]; tx12 = uvar_s[i][x1,y1,z2]
            tx21 = uvar_s[i][x1,y2,z1]; tx22 = uvar_s[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*uvar_s[i][x1,y1,z1] + (jx-x1)*uvar_s[i][x2,y1,z1]
            tx12 = (x2-jx)*uvar_s[i][x1,y1,z2] + (jx-x1)*uvar_s[i][x2,y1,z2]
            tx21 = (x2-jx)*uvar_s[i][x1,y2,z1] + (jx-x1)*uvar_s[i][x2,y2,z1]
            tx22 = (x2-jx)*uvar_s[i][x1,y2,z2] + (jx-x1)*uvar_s[i][x2,y2,z2]
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
        push!(timei_s,txyz)
    end
    push!(caltime_p,timei_p); push!(caltime_s,timei_s)
end

sum_loss_time = PyObject[]; deltt_p = PyObject[]; deltt_s = PyObject[]
for i = 1:numsta
    for j = 1:numeve
        if uobs_p[i,j] == -1
            continue
        end
        push!(sum_loss_time, qua_p[i,j]*(uobs_p[i,j]-caltime_p[i][j])^2)
        push!(deltt_p,uobs_p[i,j]-caltime_p[i][j])
    end
    for j = 1:numeve
        if uobs_s[i,j] == -1
            continue
        end
        push!(sum_loss_time, qua_s[i,j]*(uobs_s[i,j]-caltime_s[i][j])^2)
        push!(deltt_s,uobs_s[i,j]-caltime_s[i][j])
    end
end

fignum = convert(Int,ceil(sqrt(l)))
#gauss
gauss_wei = Array{Float64}(undef, 3, 3, 3)
gauss_wei[:,:,1] = [0.0012082 0.0089274 0.0012082;0.0089274 0.0659648 0.0089274;0.0012082 0.0089274 0.0012082]
gauss_wei[:,:,2] = [0.0089274 0.0659648 0.0089274;0.0659648 0.4874175 0.0659648;0.0089274 0.0659648 0.0089274]
gauss_wei[:,:,3] = [0.0012082 0.0089274 0.0012082;0.0089274 0.0659648 0.0089274;0.0012082 0.0089274 0.0012082]
filter = tf.constant(gauss_wei,shape=(3,3,3,1,1),dtype=tf.float64)

vel = tf.reshape(1 ./fvar,(1,m,n,l,1))
o_vel = (1 ./fvar)[2:m-1,2:n-1,2:l-1]
cvel = tf.nn.conv3d(vel,filter,strides = (1,1,1,1,1),padding="VALID")
n_vel = tf.reshape(cvel,(m-2,n-2,l-2))
loss1 = sum(sum_loss_time); loss2 = sum((o_vel-n_vel)^2)
loss = loss1 + loss2
#
#=laplace
laplace_wei = Array{Float64}(undef, 3, 3, 3)
laplace_wei[:,:,1] = [0 0 0; 0 1 0; 0 0 0]
laplace_wei[:,:,2] = [0 1 0; 1 -6 1; 0 1 0]
laplace_wei[:,:,3] = [0 0 0; 0 1 0; 0 0 0]
filter = tf.constant(laplace_wei,shape=(3,3,3,1,1),dtype=tf.float64)

vel = tf.reshape(1 ./fvar,(1,m,n,l,1))
cvel = tf.nn.conv3d(vel,filter,strides = (1,1,1,1,1),padding="VALID")
loss = sum(sum_loss_time) + 0.2 * sum(abs(cvel))
=#
@show run(sess,loss1)
@show run(sess,loss2)

outfolder = "/output/real/"
#=
cb = (vs,iter,loss) ->begin
    if iter % 5 == 0
        figure(figsize = (15,15))
        for i = 1:l
            subplot(fignum,fignum,i)
            plotf2 = 1.0 ./vs[1][:,:,i]
            pcolormesh(plotf2,vmin = 1/fvel0_p[1,1,i]*0.8,vmax = 1/fvel0_p[1,1,i]*1.2)
            title("layer "*string(i))
            colorbar()
        end
        savefig(folder * outfolder * "vel_$iter.png")
    end
end
BFGS!(sess,loss1;vars = [fvar],callback = cb)
#
fvar_ = run(sess,fvar)
figure(figsize = (20,20))
for i = 1:l
    subplot(fignum,fignum,i)
    plotf2 = 1.0 ./fvar_[:,:,i]
    pcolormesh(plotf2,vmin = 1/fvel0_p[1,1,i]*0.8,vmax = 1/fvel0_p[1,1,i]*1.2)
    title("layer "*string(i))
    colorbar()
end
savefig(folder * outfolder * "inversion.png")
h5write(folder * outfolder * "fvelocity_p.h5","data",fvar_)

fdeltt_p = run(sess,deltt_p); fdeltt_s = run(sess,deltt_s)
figure(figsize = (20,20)); hist(fdeltt_p,bins = bins)
xlim(-2.5,2.5); xlabel("Residual"); ylabel("Frequency")
title("histogram_s"); savefig(folder * outfolder * "hist_p_final.png")
figure(figsize = (20,20)); hist(fdeltt_s,bins = bins)
xlim(-5,5); xlabel("Residual"); ylabel("Frequency")
title("histogram_p"); savefig(folder * outfolder * "hist_s_final.png")

#
cb = (vs,iter,loss) ->begin
    if iter % 5 ==0
        figure(figsize = (20,20)); hist(vs,bins=bins)
        xlim(-2.5,2.5); xlabel("Residual"); ylabel("Frequency")
        title("histogram_p"); savefig(folder * outfolder * "zhist_P_$iter.png")
    end
end
BFGS!(sess,loss1;vars = deltt_p,callback = cb)
#
cb = (vs,iter,loss) ->begin
    if iter % 5 ==0
        figure(figsize = (20,20)); hist(vs,bins=bins)
        xlim(-5,5); xlabel("Residual"); ylabel("Frequency")
        title("histogram_s"); savefig(folder * outfolder * "zhist_S_$iter.png")
    end
end
BFGS!(sess,loss1;vars = deltt_s,callback = cb)
=#
BFGS!(sess,loss1)
@show run(sess,loss1)
@show run(sess,loss2)
@show run(sess,pvs)
