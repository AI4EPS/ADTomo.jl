using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function eikonal_three_d(u0,f,h,m,n,l,tol,verbose)
    eikonal_three_d_ = load_op_and_grad("./build/libEikonalThreeD","eikonal_three_d")
    u0,f,h,m,n,l,tol,verbose = convert_to_tensor(Any[u0,f,h,m,n,l,tol,verbose], [Float64,Float64,Float64,Int64,Int64,Int64,Float64,Bool])
    eikonal_three_d_(u0,f,h,m,n,l,tol,verbose)
end

m = 21
n = 21
l = 21
u0_ = ones(m*n*l)*1000
u0_[10 * n * l + 10 * l + 11] = 0.0
u0 = constant(u0_)
f = constant(ones(m*n*l))
h = 0.01
tol = 1e-6
verbose = true

# TODO: specify your input parameters
u = eikonal_three_d(u0,f,h,m,n,l,tol,verbose)
sess = Session(); init(sess)
u_ = run(sess, reshape(u, (21,21,21)))

# # uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(eikonal_three_d(x,f,h,m,n,l,tol,verbose)^2)
end

# TODO: change `m_` and `v_` to appropriate values

u0_init = rand(m*n*l) * 1000
# u0_init[[89,76,56,22,23,24]] .= 0.0
m_ = constant(u0_init)
v_ = rand(m*n*l)
# v_[[89,76,56,22,23,24]] .= 0.0

y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("gradtest.png")
