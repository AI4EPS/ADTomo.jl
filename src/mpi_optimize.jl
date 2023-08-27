export mpi_optimize

function mpi_optimize(_f::Function, _g!::Function, x0::Array{Float64}; 
    method::String="LBFGS", options=missing)

    r = mpi_rank()
    flag = zeros(Int64, 1)

    __cnt = 0
    __iter = 0

    function f(x)
        flag[1] = 1
        mpi_sync!(flag)
        L = _f(x)
        if r == 0 && ADCME.options.training.verbose
            __cnt += 1
            println("iter $__cnt, current loss=", L)
        end
        return L
    end

    function g!(G, x)
        if r == 0 && ADCME.options.training.verbose
            __iter += 1
            println("================== STEP $__iter ==================")
        end
        flag[1] = 2
        mpi_sync!(flag)
        _g!(G, x)
    end

    if method == "LBFGS"
        method = Optim.LBFGS(
            alphaguess=Optim.LineSearches.InitialStatic(),
            linesearch=Optim.LineSearches.BackTracking(),
        )
    elseif method == "BFGS"
        method = Optim.BFGS(
            alphaguess=Optim.LineSearches.InitialStatic(),
            linesearch=Optim.LineSearches.BackTracking(),
        )
    else
        error("Method $method not implemented.")
    end

    options = coalesce(options, Optim.Options())

    r = mpi_rank()
    if r == 0
        println("[MPI Size = $(mpi_size())] Optimization starts...")
        result = Optim.optimize(f, g!, x0, method, options)
        flag[1] = 0
        mpi_sync!(flag)
        return result
    else
        while true
            mpi_sync!(flag)
            if flag[1] == 1
                _f(x0)
            elseif flag[1] == 2
                _g!(zero(x0), x0)
            else
                break
            end
        end
        return nothing
    end

end


# https://github.com/kailaix/ADCME.jl/blob/074c84443cfe89b66a1b8900a83d60f81d4fbc03/src/optim.jl#L792C36-L792C52
function mpi_optimize(sess::PyObject, loss::PyObject, 
    vars::Union{Array{PyObject},PyObject,Missing}=missing,
    grads::Union{Array{T},Nothing,PyObject,Missing}=missing;
    method::String="LBFGS", options=missing) where T<:Union{Nothing, PyObject}

    vars = coalesce(vars, get_collection())
    grads = coalesce(grads, gradients(loss, vars))
    if isa(vars, PyObject)
        vars = [vars]
    end
    if isa(grads, PyObject)
        grads = [grads]
    end
    if length(grads) != length(vars)
        error("ADCME: length of grads and vars do not match")
    end

    if !all(is_variable.(vars))
        error("ADCME: the input `vars` should be trainable variables")
    end

    idx = ones(Bool, length(grads))
    pynothing = pytypeof(PyObject(nothing))
    for i = 1:length(grads)
        if isnothing(grads[i]) || pytypeof(grads[i]) == pynothing
            idx[i] = false
        end
    end
    grads = grads[idx]
    vars = vars[idx]
    sizes = []
    for v in vars
        push!(sizes, size(v))
    end
    grds = vcat([tf.reshape(g, (-1,)) for g in grads]...)
    vs = vcat([tf.reshape(v, (-1,)) for v in vars]...)
    x0 = run(sess, vs)
    pl = placeholder(x0)
    n = 0
    assign_ops = PyObject[]
    for (k, v) in enumerate(vars)
        vnew = tf.reshape(pl[n+1:n+prod(sizes[k])], sizes[k])
        vnew = cast(vnew, get_dtype(pl))
        push!(assign_ops, assign(v, vnew))
        n += prod(sizes[k])
    end

    function _f(x)
        run(sess, assign_ops, pl => x)
        run(sess, loss)
    end

    function _g!(G, x)
        run(sess, assign_ops, pl => x)
        G[:] = run(sess, grds)
    end

    result = mpi_optimize(_f, _g!, x0, method=method, options=options)
    if mpi_rank() == 0
        x = result.minimizer
        x = run(sess, assign_ops, pl => x)
        l = result.minimum
        return x
    else
        return nothing
    end

end