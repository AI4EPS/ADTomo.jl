using ADCME
using ADTomo
using PyCall
using LinearAlgebra
using PyPlot
using Random
using Optim
Random.seed!(233)

mpi_init()
rank = mpi_rank()
nproc = mpi_size()

reset_default_graph()
#include("eikonal_op.jl")

m = 21
n = 21
l = 21
h = 1.0

f1 = ones(m,n,l)
f1[5:8, 5:8, 5:8] .= 2.

f2 = ones(m,n,l)
f2[5:8, 5:8, 5:8] .= 2.

u1 = 1000 * ones(m,n,l)
u1[10, 10, 10] = 0

u2 = 1000 * ones(m,n,l)
u2[10, 10, 10] = 0

uobs1_ = eikonal3d(u1,f1,h,m,n,l,1e-6,false)
uobs2_ = eikonal3d(u2,f2,h,m,n,l,1e-6,false)

sess = Session()
uobs1 = run(sess, uobs1_)
uobs2 = run(sess, uobs2_)

# fvar1_ = Variable(ones(m, n, l))
# fvar2_ = Variable(ones(m, n, l))

# # option 1
# fvar_ = vcat(tf.reshape(fvar1_, (-1,)), tf.reshape(fvar2_, (-1,)))
# fvar = mpi_bcast(fvar_)

# fvar1 = tf.reshape(fvar[1:prod(size(fvar1_))], size(fvar1_))
# fvar2 = tf.reshape(fvar[prod(size(fvar1_))+1:end], size(fvar2_))

# option 2
# fvar1 = mpi_bcast(fvar1_)
# fvar2 = mpi_bcast(fvar2_)

# option 3
fvar_ = Variable(ones(m, n, l*2))
# fvar_ = Variable(ones(m*2, n, l))
fvar = mpi_bcast(fvar_)
# fvar1 = fvar[1:m,:,:]
# fvar2 = fvar[m+1:end,:,:]
fvar1 = fvar[:,:,1:l]
fvar2 = fvar[:,:,l+1:end]

uvar1 = eikonal3d(u1,fvar1,h,m,n,l,1e-6,false)
uvar2 = eikonal3d(u2,tf.multiply(fvar1, fvar2),h,m,n,l,1e-6,false)
# uvar2 = eikonal3d(u2,fvar2,h,m,n,l,1e-6,false)

loss_ = sum((uobs1-uvar1)^2) + sum((uobs2-uvar2)^2)
loss = mpi_sum(loss_) 

init(sess)
@show run(sess, loss_)
@show run(sess, loss)

options = Optim.Options(iterations = 100)
result = ADTomo.mpi_optimize(sess, loss, method="LBFGS", options = options,  steps = 10001)

if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]

    fvar1 = result[1][:,:,1:l]
    # fvar1 = result[1][1:m,:,:]

    close("all")
    pcolormesh(f1[6,:,:])
    colorbar()
    savefig("Exact.png")

    close("all")
    pcolormesh(fvar1[6,:,:])
    colorbar()
    savefig("Estimate.png")


    close("all")
    pcolormesh(abs.(fvar1[6,:,:]-f1[6,:,:]))
    colorbar()
    savefig("Diff.png")

    fvar2 = result[1][:,:,l+1:end]
    # fvar2 = result[1][m+1:end,:,:]

    close("all")
    pcolormesh(f2[6,:,:])
    colorbar()
    savefig("Exact2.png")

    close("all")
    pcolormesh(fvar2[6,:,:])
    colorbar()
    savefig("Estimate2.png")

    close("all")
    pcolormesh(fvar1[6,:,:].*fvar2[6,:,:]-f2[6,:,:])
    # pcolormesh(abs.(fvar2[6,:,:]-f2[6,:,:]))
    colorbar()
    savefig("Diff2.png")


end

mpi_finalize()