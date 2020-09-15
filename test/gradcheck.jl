function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end


function gradcheck(f, xs...; tol=1e-5)
    ng = ngradient(f, xs...)
    g = collect(grad(f, xs...)[2])
    for i in 1:length(ng)
        eq = isapprox(ng[i], g[i], rtol=tol, atol=tol)
        # println(eq)
        if !eq
            return false
        end
    end
    return true
end


function check_convergence(f, args...; p=1, verbose=false, lr=1e-4, epochs=100)
    opt = Adam(lr=1e-4)
    # run once to overcome possible initialization issues
    # _, g = grad(f, args...)
    # update!(opt, args[p], g[p])
    # calculate loss at the beginning
    L0 = f(args...)
    L = L0
    for i=1:epochs
        L, g = grad(f, args...)
        if verbose
            println(L)
        end
        update!(opt, args[p], g[p])
    end
    return L < L0
end
