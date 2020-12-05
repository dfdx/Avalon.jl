using Avalon
import Avalon.fit!
using Distributions
using MLDataUtils
using MLDatasets
import Plots


################################################################################
#                                Coupling                                      #
################################################################################


mutable struct Coupling
    mask  # {0, 1} array for input splitting
    s     # callable struct representing s function (scale)
    t     # callable struct representing t function (translation)
end


Base.show(io::IO, c::Coupling) = print(io, "Coupling($(size(c.mask)), $(typeof(c.s)), $(typeof(c.t)))")


"""
Forward mapping x → z. Same as f() function in the paper
"""
function fwd_map(c::Coupling, x)
    mask, s, t = c.mask, c.s, c.t
    x1 = x .* mask
    sv = s(x1)
    tv = t(x1)
    # z_1:d = x_1:d
    # z_d+1:D = exp(s(x_1:d)) * x_d+1:D + m(x_1:d)
    y = x1 .+ (1f0 .- mask) .* (x .* exp.(sv) .+ tv)
    log_det_J = sum(sv; dims=1)
    return y, log_det_J
end


"""
Inverse mapping z → x. Same as g() function in the paper
"""
function inv_map(c::Coupling, z)
    x = z
    mask, s, t = c.mask, c.s, c.t
    x1 = x .* mask
    sv = s(x1)
    tv = t(x1)
    # x_1:d = z_1:d
    # x_d+1:D = z_d+1:D - m(z_1:d) * exp(-s(z_1:d))
    y = x1 .+ (1f0 .- mask) .* ((x .- tv) .* exp.(-sv))
    return y
end


################################################################################
#                                RealNVP                                       #
################################################################################

make_s(xz_len, u_len) =
    Sequential(
        Linear(xz_len, u_len),
        x -> relu.(x),
        Linear(u_len, u_len),
        x -> relu.(x),
        Linear(u_len, xz_len),
        x -> tanh.(x)
    )

make_t(xz_len, u_len) =
    Sequential(
        Linear(xz_len, u_len),
        x -> relu.(x),
        Linear(u_len, u_len),
        x -> relu.(x),
        Linear(u_len, xz_len)
    )


mutable struct RealNVP
    prior::MvNormal
    c1::Coupling
    c2::Coupling
    c3::Coupling
    c4::Coupling
    c5::Coupling
    c6::Coupling
end

Base.show(io::IO, m::RealNVP) = print(io, "RealNVP($(m.c1), $(m.c2))")


function RealNVP(xz_len::Int, u_len::Int)
    mask = vcat([1 for i=1:Int(xz_len / 2)], [0 for i=1:Int(xz_len / 2)])
    neg_mask = [1 - x for x in mask]
    prior = MvNormal(zeros(xz_len), ones(xz_len))  # note: not CuArray friendly right now
    c1 = Coupling(mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    c2 = Coupling(neg_mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    c3 = Coupling(mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    c4 = Coupling(neg_mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    c5 = Coupling(mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    c6 = Coupling(neg_mask, make_s(xz_len, u_len), make_t(xz_len, u_len))
    RealNVP(prior, c1, c2, c3, c4, c5, c6)
end


function fwd_map(flow::RealNVP, x)
    z = x
    z, log_det_J1 = fwd_map(flow.c1, z)
    z, log_det_J2 = fwd_map(flow.c2, z)
    z, log_det_J3 = fwd_map(flow.c3, z)
    z, log_det_J4 = fwd_map(flow.c4, z)
    z, log_det_J5 = fwd_map(flow.c5, z)
    z, log_det_J6 = fwd_map(flow.c6, z)
    logp = log_det_J1 + log_det_J2 + log_det_J3 + log_det_J4 + log_det_J5 + log_det_J6
    return z, logp
end


function inv_map(flow::RealNVP, z)
    x = z
    x = inv_map(flow.c6, x)
    x = inv_map(flow.c5, x)
    x = inv_map(flow.c4, x)
    x = inv_map(flow.c3, x)
    x = inv_map(flow.c2, x)
    x = inv_map(flow.c1, x)
    return x
end


function Distributions.gradlogpdf(ds::AbstractVector, d::MvNormal, x::AbstractMatrix)
    ret = similar(x)
    for j in 1:size(x, 2)
        ret[:, j] = Distributions.gradlogpdf(d, @view x[:, j]) .* ds[j]
    end
    return ret
end

@nodiff Distributions.logpdf(_d::MvNormal, _x) _d
@diffrule Distributions.logpdf(_d::MvNormal, _x) _x Distributions.gradlogpdf(dy, _d, _x)


function logprob(flow::RealNVP, x)
    z, logp = fwd_map(flow, x)
    return Distributions.logpdf(flow.prior, z) .+ logp
end


function loss(flow::RealNVP, x)
    return -mean(logprob(flow, x))
end


function fit!(flow::RealNVP, X::AbstractMatrix{T};
                        n_epochs=50, batch_size=100, report_every=1, opt=SGD(1e-4)) where T
    for epoch in 1:n_epochs
        epoch_cost = 0
        t = @elapsed for (i, x) in enumerate(eachbatch(X, size=batch_size))
            cost, g = grad(loss, flow, x)
            update!(opt, flow, g[1]; ignore=[(c, :mask) for c in [:c1, :c2, :c3, :c4, :c5, :c6]])
            epoch_cost += cost
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_cost=$(epoch_cost / (size(X,2) / batch_size)), elapsed=$t")
        end
    end
    return flow
end


## Example data

Plots.gr()


function make_moons(;n_samples=1000)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(LinRange(0, pi, n_samples_out))
    outer_circ_y = sin.(LinRange(0, pi, n_samples_out))
    inner_circ_x = 1 .- cos.(LinRange(0, pi, n_samples_out))
    inner_circ_y = 1 .- sin.(LinRange(0, pi, n_samples_out)) .- 0.5

    X = [outer_circ_x outer_circ_y; inner_circ_x  inner_circ_y]'
    X .+= 0.1 * randn(size(X))
    y = [zeros(Int, n_samples_out); ones(Int, n_samples_in)]
    return X, y
end


function main()
    X, y = make_moons(n_samples=200)
    flow = RealNVP(2, 256)
    fit!(flow, X, report_every=100, n_epochs=1000, opt=Adam(lr=1e-4))

    # map moon-disributed X to normally distributed Z
    Z, _ = fwd_map(flow, X)
    Plots.scatter(X[1, :], X[2, :])
    Plots.scatter!(Z[1, :], Z[2, :])

    # map normally distributed Z to moon-distributed X
    Z2 = rand(MvNormal(zeros(2), ones(2)), 200)
    X2 = inv_map(flow, Z2)
    Plots.scatter!(Z[1, :], Z[2, :])
    Plots.scatter(X[1, :], X[2, :])
end
