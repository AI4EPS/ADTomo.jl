using ADCME
using ADTomo
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)


reset_default_graph()
#include("eikonal_op.jl")

m = 51
n = 51
l = 51
h = 0.01

f = ones(m,n,l)
f[5:8, 5:8, 5:8] .= 2.

u0 = 1000 * ones(m,n,l)
u0[10, 10, 10] = 0

h = 5

u = eikonal3d(u0,f,h,m,n,l,1e-6,false)

sess = Session()
uobs = run(sess, u)

fvar = Variable(ones(m, n, l))
uvar = eikonal3d(u0,fvar,h,m,n,l,1e-6,false)

loss = sum((u-uvar)^2)

init(sess)
@show run(sess, loss)

BFGS!(sess,loss)

fvar_ = run(sess, fvar)

close("all")
pcolormesh(f[6,:,:])
colorbar()
savefig("Exact.png")

close("all")
pcolormesh(fvar_[6,:,:])
colorbar()
savefig("Estimate.png")


close("all")
pcolormesh(abs.(fvar_[6,:,:]-f[6,:,:]))
colorbar()
savefig("Diff.png")
