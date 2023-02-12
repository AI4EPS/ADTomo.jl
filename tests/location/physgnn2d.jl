using ADCME
using ADEikonal
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

reset_default_graph()



m = 60
n = 30
h = 0.1

f = ones(n+1, m+1)

dx = 5
dy = 5

# srcx = 1:dx:m
# srcy = 5*ones(Int64, length(1:dx:m))
srcx = [15, 45]
srcy = [5, 5]

recx = collect([20, 30, 40, 50])
recy = collect([15, 25, 20, 10])

u = PyObject[]
t = PyObject[]
magn = []

xs = zeros(length(recx), n+1, m+1)
ys = zeros(length(recx), n+1, m+1)
for i = 1:n+1
    for j = 1:m+1
        xs[:, i, j] .= (j-1)*h
        ys[:, i, j] .= (i-1)*h
    end
end

σ = 1.0 *h
for (sx, sy) in zip(srcx, srcy)

    u_ = eikonal(f,sx,sy,h)
    push!(u, u_)
   
    magn_ = @. 1. /(2π * σ) * exp( - 0.5 * (((xs-recx*h)/σ)^2 + ((ys-recy*h)/σ)^2)) /σ*h^2
    t_ = sum(u_ .* magn_, dims=(2,3))
    push!(magn, magn_)
    push!(t, t_)

end


sess = Session()
tobs = run(sess, t)
uobs = run(sess, u)

@show tobs

figure()
pcolormesh(uobs[1])
colorbar()
savefig("obs.png")

figure()
pcolormesh(magn[1][1,:,:])
colorbar()
savefig("magn.png")

## 
t = PyObject[]
u = PyObject[]
magn = PyObject[]

recx = Variable([50.0, 50.0, 50.0, 50.0])
recy = Variable([5.0, 5.0, 5.0, 5.0])
recx_ = reshape(recx, [length(recx), 1, 1])
recy_ = reshape(recy, [length(recy), 1, 1])

for (sx, sy) in zip(srcx, srcy)

    u_ = eikonal(f,sx,sy,h)
    push!(u, u_)

    magn_ = 1. /(2π * σ) * exp( - 0.5 * (((xs-recx_*h)/σ)^2 + ((ys-recy_*h)/σ)^2)) /σ*h^2
    t_ = sum(u_ .* magn_, dims=(2,3))
    push!(t, t_)
    push!(magn, magn_)

end

loss = sum(sum(abs.(t - tobs)))

init(sess)
@show run(sess, loss)

# error()
# figure()
# pcolormesh(run(sess, magn))
# colorbar()
# savefig("magn_inv.png")

# error()

# lineview(sess, F, loss, f, ones(n+1, m+1))
# gradview(sess, pl, loss, ones((n+1)* (m+1)))


# meshview(sess, pl, loss, F0'[:])

BFGS!(sess, loss, 200, var_to_bounds=Dict(recx=>(5.0,m-5.0), recy=>(5.0,n-5.0)))

magn_ = run(sess, magn)[1]
for i in 1:length(recx)
    figure()
    pcolormesh(magn_[i,:,:])
    colorbar()
    savefig("magn_inv_$i.png")
end

# figure(figsize=(10, 4))
# subplot(121)
# pcolormesh(f)
# colorbar()
# title("True")
# subplot(122)
# pcolormesh(run(sess, F))
# colorbar()
# title("Inverted")
# savefig("inversion.png")


sess = Session(); init(sess)

env = Environment(sess)
sim = Simulator(sess)

loss = empirical_sinkhorn(env.obs, sim.obs, method = "lp")

opt = AdamOptimizer(0.001).minimize(loss)
init(sess)
@info run(sess, loss, env.μ=>sample_exact(env),
    sim.μ=>sample_latent(sim))

for i = 1:1000
    _, l = run(sess, [opt, loss], env.μ=>sample_exact(env),
                     sim.μ=>sample_latent(sim))

    if mod(i, 50)==0
        for k = 1:100
            d = run(sess, sim.dnn, sim.μ=>sample_latent(sim))
        end

        @info i, l  
    end
end