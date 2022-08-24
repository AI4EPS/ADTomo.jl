using ADCME
using ADEikonal
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

################## Simulation ##################
reset_default_graph()

m = 60
n = 30
h = 0.1
dx = 5
dy = 5

vel = ones(n+1, m+1)

srcx = [15, 45, 15, 45]
srcy = [15, 15, 45, 45]
srcz = [1, 1, 1, 1]

# recx = collect(Float64, [20,])
# recy = collect(Float64, [20,])
# recz = collect(Float64, [16,])
recx = collect(Float64, [20, 30, 40, 20, 30, 40, 20, 30, 40,])
recy = collect(Float64, [20, 20, 20, 30, 30, 30, 40, 40, 40,])
recz = collect(Float64, [16, 17, 18, 19, 20, 21, 22, 23, 24,])

t_ = PyObject[]

xs = zeros(length(recx), n+1, m+1)
ys = zeros(length(recx), n+1, m+1)
for i = 1:n+1
    for j = 1:m+1
        xs[:, i, j] .= (j-1)*h
        ys[:, i, j] .= (i-1)*h
    end
end

σ = 1.0 * h
for (sx, sy, sz) in zip(srcx, srcy, srcz)

    u = eikonal(vel,1,sz,h)

    R = @. sqrt((recx - sx)^2 + (recy - sy)^2)
    magn = @. 1. /(2π * σ) * exp( - 0.5 * (((xs-R*h)/σ)^2 + ((ys-recz*h)/σ)^2)) /σ*h^2

    t = sum(u .* magn, dims=(2,3))

    push!(t_, t)

end


sess = Session()
tobs = run(sess, t_)

################## Inversion ##################
t_ = PyObject[]

recx_ = Variable(ones(length(recx)) .* 10.0)
recy_ = Variable(ones(length(recx)) .* 10.0)
recz_ = Variable(ones(length(recx)) .* 10.0)
# recx_ = Variable(recx)
# recy_ = Variable(recy)
# recz_ = Variable(recz)
recx = reshape(recx_, [length(recx_), 1, 1])
recy = reshape(recy_, [length(recy_), 1, 1])
recz = reshape(recz_, [length(recz_), 1, 1])

for (sx, sy, sz) in zip(srcx, srcy, srcz)

    u = eikonal(vel,1,sz,h)

    R = sqrt((recx - sx)^2 + (recy - sy)^2)
    magn = 1. /(2π * σ) * exp( - 0.5 * (((xs-R*h)/σ)^2 + ((ys-recz*h)/σ)^2)) /σ*h^2
    t = sum(u .* magn, dims=(2,3))

    push!(t_, t)

end

loss = sum(sum(abs.(t_ - tobs)))

init(sess)
@show "Init loss = ", run(sess, loss)

BFGS!(sess, loss, 200, var_to_bounds=Dict(recx_=>(5.0,m-5.0), recy_=>(5.0,m-5.0), recz_=>(5.0,n-5.0)))
@show run(sess, [recx_, recy_, recz_])
