# GV post filters
# Reference:
# Silén, Hanna, et al. "Ways to Implement Global Variance in Statistical Speech
# Synthesis." INTERSPEECH. 2012.

immutable VarianceScaling
    σ²::Vector{Float64}
end

function fvpostf!(vs::VarianceScaling, src::AbstractMatrix)
    const D = size(src, 1)
    μ = mean(src, 2)
    src[:,:] = sqrt(vs.σ² ./ var(src[1:D,:], 2)) .* (src .- μ) .+ μ
    nothing
end

function fvpostf(vs::VarianceScaling, src::AbstractMatrix)
    filtered = copy(src)
    fvpostf!(vs, filtered)
    filtered
end
