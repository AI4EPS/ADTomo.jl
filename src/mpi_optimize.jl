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