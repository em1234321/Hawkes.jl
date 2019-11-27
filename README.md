# Hawkes.jl

Simulation and likelihood methods for univariate and multivariate Hawkes Processes with exponential kernels

## Example

```julia
using Hawkes
using DynamicHMC, LogDensityProblems, TransformVariables
using Distributions, Parameters, Random, Statistics
import ForwardDiff

# Simulate a 2-dimensional Hawkes process
u = [0.5, 0.1]
α = [1.3 0.8; 0.0 1.3]
δ = [2.0, 2.0]
ts = hawkes_simulate(u, α, δ, 10^4)

# Use DynamicHMC to recover the parameters from the simulated times
struct HP
    ts::Vector{Vector{Float64}}
end

function (problem::HP)(θ)
    @unpack u1,u2,α1,α2,δ1 = θ
    @unpack ts = problem
    u = [u1, u2]
    α = [α1 α2; 0.0 α1]
    δ = [δ1, δ1]
    prior = loglikelihood(Exponential(10.0), [u1, u2, α1, α2, δ1])
    hawkes_loglikelihood(u,α,δ,ts) + prior
end

p = HP(ts)
tr = as((u1 = as_positive_real, u2 = as_positive_real, α1 = as_positive_real, α2 = as_positive_real, δ1 = as_positive_real))
P = TransformedLogDensity(tr, p)
dP = ADgradient(:ForwardDiff, P);

results = mcmc_with_warmup(Random.MersenneTwister(1), dP, 10^4)

transform.(tr, results.chain) |> x -> collect(zip(keys(x[1]),  mapslices(mean,  map(collect,x), dims=[1])))

5-element Array{Tuple{Symbol,Float64},1}:
 (:u1, 0.49881371330080354)
 (:u2, 0.09750867941083316)
 (:α1, 1.241428210846674)
 (:α2, 0.765949249773491)
 (:δ1, 1.8858744824012086)

```

## References

This implementation is based on the following two papers:

- [Exact simulation of Hawkes process with exponentially decaying intensity](https://projecteuclid.org/euclid.ecp/1465315601)
- [Hawkes Processes: Simulation, Estimation, and Validation](https://pat-laub.github.io/pdfs/honours_thesis.pdf)


