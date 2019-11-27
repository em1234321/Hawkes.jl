module Hawkes

export hawkes_simulate, hawkes_loglikelihood

"""
Simulate a univariate Hawkes process with intensity:

`λ(t) = α + sum_{t_k<t}{α*exp(δ*(t-t_k))}`
# Arguments
- `u::Float64`: background intensity
- `α::Float64`: arrival intensity jump-size
- `δ::Float64`: exponential decay rate
- `n::Int`: number of times to simulate
"""
function hawkes_simulate(u::Float64, α::Float64, δ::Float64, n::Int)::Vector{Float64}
    λ = u
    t = 0.0
    ts = Vector{Float64}(undef, n)
    for i in 1:n
        D = 1.0 + δ*log(rand())/(λ - u)
        dt1 = -log(rand())/u
        dt2 = min(dt1, -log(max(0.0,D))/δ)
        dt = ifelse(D < 0.0, dt1, dt2)
        t += dt
        ts[i] = t
        λ = α + (λ - u)*exp(-δ*dt) + u
    end
    ts
end

"""
Simulate a multivariate Hawkes process with intensities:

`λ(t)_i = α_i + sum_{t_k<t}{α_(i,j)*exp(δ_i*(t-t_k))}`
# Arguments
- `u::AbstractVector{Float64}`: background intensities
- `α::AbstractMatrix{Float64}`: arrival intensity jump-sizes
- `δ::AbstractVector{Float64}`: exponential decay rates
- `n::Int`: number of times to simulate
"""
function hawkes_simulate(u::AbstractVector{Float64}, α::AbstractMatrix{Float64}, δ::AbstractVector{Float64}, n::Int)::Vector{Vector{Float64}}
    t = 0.0
    M = length(u)
    λ = copy(u)
    ts = [Vector{Float64}() for _ in 1:M]
    for i in 1:n
        dt_min = Inf
        dt_min_m = 0
        for m in 1:M
            D = 1.0 + δ[m]*log(rand())/(λ[m] - u[m])
            dt1 = -log(rand())/u[m]
            dt2 = min(dt1, -log(max(0.0,D))/δ[m])
            dt = ifelse(D < 0.0, dt1, dt2)
            if dt < dt_min
                dt_min = dt
                dt_min_m = m
            end
        end
        t += dt_min
        push!(ts[dt_min_m], t)
        for m in 1:M
            λ[m] = α[m,dt_min_m] + (λ[m] - u[m])*exp(-δ[m]*dt_min) + u[m]
        end
    end
    ts
end

# Real types are used rather than Float64 so that automatic differentiation methods will work.
"""
Calculate the log-likelihood for a univariate Hawkes process with intensity:

`λ(t) = α + sum_{t_k<t}{α*exp(δ*(t-t_k))}`
# Arguments
- `u::Real`: background intensity
- `α::Real`: arrival intensity jump-size
- `δ::Real`: exponential decay rate
- `ts::AbstractVector{Real}`: arrival times
"""
function hawkes_loglikelihood(u::Real, α::Real, δ::Real, ts::AbstractVector)::Real
    ll1 = 0.0
    ll2 = 0.0
    t = 0.0
    λ = u
    N = length(ts)
    T = ts[N]
    A = 0.0
    for i in 1:N
        dt = ts[i] - t
        t = ts[i]
        ll1 += log(u + α*A)
        ll2 += -1.0 + exp(-δ*(T-t))
        dtn = ts[min(N,i+1)] - ts[i]
        A = exp(-δ*dtn)*(1.0 + A)
    end
    ll1 + α*ll2/δ - u*T
end

# Real types are used rather than Float64 so that automatic differentiation methods will work.
"""
Calculate the log-likelihood for a multivariate Hawkes process with intensities:


`λ(t)_i = α_i + sum_{t_k<t}{α_(i,j)*exp(δ_i*(t-t_k))}`
# Arguments
- `u::AbstractVector{Real}`: background intensities
- `α::AbstractMatrix{Real}`: arrival intensity jump-sizes
- `δ::AbstractVector{Real}`: exponential decay rates
- `ts::AbstractVector{AbstractVector{Real}}`: arrival times
"""
function hawkes_loglikelihood(u::AbstractVector{R},α::AbstractMatrix{R},δ::AbstractVector{R},ts::AbstractVector)::R where R<:Real
    res = 0.0
    M = length(ts)
    T = maximum(last.(ts))
    for m in 1:M
        s = 0.0
        Rdiag = zeros(R, length(ts[m]))
        Rnondiag = zeros(R, length(ts[m]))
        ix = ones(Int, M)
        for i in 2:length(ts[m])
            dt = ts[m][i] - ts[m][i-1]
            Rdiag[i] = (1.0 + Rdiag[i-1])*exp(-δ[m]*dt)
            Rnondiag[i] = Rnondiag[i-1]*exp(-δ[m]*dt)
            for n in 1:M
                m == n && continue
                for j in ix[n]:length(ts[n])
                    if ts[n][j] >= ts[m][i-1]
                        if ts[n][j] < ts[m][i]
                            Rnondiag[i] += exp(-δ[m]*(ts[m][i] - ts[n][j]))
                        else
                            ix[n] = j
                            break
                        end
                    end
                end
            end
        end
        for n in 1:M
            for j in 1:length(ts[n])
                s += α[m,n]*(1.0 - exp(-δ[m]*(T - ts[n][j])))/δ[m]
            end
        end
        res += -u[m]*T - s
        for i in 1:length(ts[m])
            s = u[m]
            for n in 1:M
                s += α[m,n]*ifelse(m==n, Rdiag[i], Rnondiag[i])
            end
            res += log(s)
        end
    end
    res
end

end # module
