using HDF5

region = "demo/"
folder = "../local/"*region*"readin_data/"
rfile = open(folder*"range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile));

vel_p = Dict(); fvel0_p = ones(m,n,l); vel0_p = ones(m,n,l) 
vel_s = Dict(); fvel0_s = ones(m,n,l); vel0_s = ones(m,n,l)

vel_h = [0,1,3,4,5,17,25]
vel_p = [3.20,4.50,4.80,5.51,6.21,6.89,7.83]
vel_s = [1.50,2.40,2.78,3.18,3.40,3.98,4.52]

nl = 1; nvel = vel_p[nl]
for i = 1:l
    if nl < 7 && (i-dz)*h >= vel_h[nl+1]
        global nl += 1
        global nvel = vel_p[nl]
    end
    fvel0_p[:,:,i] .= 1/nvel
    vel0_p[:,:,i] .= nvel
end
nl = 1; nvel = vel_s[nl]
for i = 1:l
    if nl < 7 && (i-dz)*h >= vel_h[nl+1]
        global nl += 1
        global nvel = vel_s[nl]
    end
    fvel0_s[:,:,i] .= 1/nvel
    vel0_s[:,:,i] .= nvel
end
if !isdir(folder*"velocity/") mkdir(folder*"velocity/") end
h5write(folder * "velocity/GIL7_vel0_p.h5","data",vel0_p)
h5write(folder * "velocity/GIL7_vel0_s.h5","data",vel0_s)