using HDF5
using PyPlot

v = h5read("readin_data/store/new4/2/inv_S_0.1/intermediate/post_101.h5","data")
for i = 1:82
    local fig = figure(figsize=(10,2))
    local ax = fig[:add_subplot](111)
    local pcm = ax[:pcolormesh](transpose(v[i,:,2:10]),cmap="turbo_r")
    colorbar(pcm)
    ax.invert_yaxis()
    savefig("readin_data/store/new4/2/vertical/S_all_x/$i.png")
    close()
end