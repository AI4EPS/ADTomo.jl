using ADCME
using ADEikonal
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

reset_default_graph()
#include("eikonal_op.jl")

m = 60
n = 30
h = 0.1

f = ones(n+1, m+1)
f[12:18, 32:48] .= 2.

srcx = 40
srcy = 15

dx = 5
dy = 5

u = PyObject[]

#push!(u, eikonal(f,srcx,srcy,h))

for (k,(x,y)) in enumerate(zip(5*ones(Int64, length(1:dx:n)), 1:dx:n))
    push!(u,eikonal(f,x,y,h))
end

for (k,(x,y)) in enumerate(zip(55*ones(Int64, length(1:dx:n)), 1:dx:n))
    push!(u,eikonal(f,x,y,h))
end

for (k,(x,y)) in enumerate( zip(1:dy:m,5*ones(Int64, length(1:dy:m))))
    push!(u,eikonal(f,x,y,h))
end

for (k,(x,y)) in enumerate( zip(1:dy:m,25*ones(Int64, length(1:dy:m))))
    push!(u,eikonal(f,x,y,h))
end


sess = Session()
uobs = run(sess, u)

F = Variable(ones(n+1, m+1))
#F = Variable(f)
# pl = placeholder(F0'[:])
# F = reshape(pl, n+1, m+1)

# F = Variable(ones(n+1, m+1))
u = PyObject[]

#push!(u,eikonal(F,srcx,srcy,h))

for (k,(x,y)) in enumerate(zip(5*ones(Int64, length(1:dx:n)), 1:dx:n))
    push!(u,eikonal(F,x,y,h))
end

for (k,(x,y)) in enumerate(zip(55*ones(Int64, length(1:dx:n)), 1:dx:n))
    push!(u,eikonal(F,x,y,h))
end

for (k,(x,y)) in enumerate(zip(1:dy:m,5*ones(Int64, length(1:dy:m))))
    push!(u,eikonal(F,x,y,h))
end

for (k,(x,y)) in enumerate(zip(1:dy:m,25*ones(Int64, length(1:dy:m))))
    push!(u,eikonal(F,x,y,h))
end

# loss = sum([sum((uobs[i][5:5:end,55] - u[i][5:5:end,55])^2) for i = 1:length(u)])
# loss = sum([sum((uobs[i][1:end,55] - u[i][1:end,55])^2) for i = 1:length(u)])
loss = sum([sum((uobs[i][1:end,55] - u[i][1:end,55])^2) + 
            sum((uobs[i][1:end,5] - u[i][1:end,5])^2) +
            sum((uobs[i][5,1:end] - u[i][5,1:end])^2) +
            sum((uobs[i][25,1:end] - u[i][25,1:end])^2) for i = 1:length(u)])

init(sess)
@show run(sess, loss)

# lineview(sess, F, loss, f, ones(n+1, m+1))
# gradview(sess, pl, loss, ones((n+1)* (m+1)))


# meshview(sess, pl, loss, F0'[:])

BFGS!(sess, loss, 200,var_to_bounds=Dict(F=>(0.5,100.0)))
#BFGS!(sess, loss, 200,var_to_bounds=Dict(F=>(0.5,100.0)))

pcolormesh(run(sess, F))
colorbar()
savefig("inversion.png")
