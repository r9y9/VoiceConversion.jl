# Alignment functions to create parallel data

# align_mcep performs dtw-based mel-cesptrum feature alignment between two
# speakers.
function align_mcep(src::Matrix{Float64}, # feature matrix of source speaker
                    tgt::Matrix{Float64}; # feature matrix of target speaker
                    th::Float64=14.0,     # threshold to cut off silence frames
                    alpha::Float64=0.35,  # all-pass constant
                    framelen::Int=1024
                    )
    @assert size(src, 1) == size(tgt, 1) ||
        error("order of feature vector must be equal")

    # Alignment
    d = DTW(fstep=0, bstep=2) # allow one skip
    path = fit!(d, src, tgt)

    # create aligned tgt
    newtgt = zeros(eltype(src), size(src))
    newtgt[:,path] = tgt[:,1:length(path)]

    # interpolation
    # TODO(ryuichi) better solution
    hole = setdiff([path[1]:path[end]], path)
    for i in hole
        if i > 1 && i < size(src, 2)
            newtgt[:,i] =
                (newtgt[:,i-1] + newtgt[:,i+1]) / 2.0
        end
    end

    # Remove silence segment
    E = log(mc2e(src, alpha, framelen))
    src = src[:, E .> th]
    newtgt = newtgt[:, E .> th]

    return src, newtgt
end
