using HDF5

folder = "../readin_data/discrete/len_2_4/"
file = folder * "step_1/intermediate_P/iter_30.h5"
fvel = h5read(file,"data")

rfile = open("readin_data/range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile));

vel0_p = ones(m,n,l)
for i = 1:l
    vel0_p[:,:,i] .= 1/fvel[i]
end
h5write(folder * "step_2/for_P/1D_vel0_p.h5","data",vel0_p)