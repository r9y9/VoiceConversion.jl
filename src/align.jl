# Alignment functions to create parallel data

using .DTWs

import MelGeneralizedCepstrums: mc2e

# general feature alignment
function align(src::AbstractMatrix, # source feature matrix
               tgt::AbstractMatrix  # target feature matrix
               )
    if size(src, 1) != size(tgt, 1)
        throw(DimentionMismatch("order of feature vector must be equal"))
    end

    # Alignment
    d = DTW(fstep=0, bstep=2) # allow one skip
    path = fit!(d, src, tgt)

    # create aligned tgt
    newtgt = zeros(eltype(src), size(src))
    newtgt[:,path] = tgt[:,1:length(path)]

    # interpolation
    # TODO(ryuichi) better solution
    hole = setdiff(collect(path[1]:path[end]), path)
    for i in hole
        if i > 1 && i < size(src, 2)
            for j=1:size(newtgt, 1)
                @inbounds newtgt[j,i] = (newtgt[j,i-1] + newtgt[j,i+1]) / 2.0
            end
        end
    end

    src, newtgt
end

# alignment function specialized for mel-cepstrum
function align_mcep(src::AbstractMatrix,       # source feature matrix
                    tgt::AbstractMatrix,       # target feature matrix
                    α::AbstractFloat,         # all-pass constant
                    fftlen::Integer;           # fft length
                    threshold::Float64=-14.0,  # threshold to remove silence frames
                    remove_silence::Bool=true
                    )
    src, newtgt = align(src, tgt)

    if remove_silence
        e = mc2e(src, α, fftlen)
        E = log(e)
        src = src[:, E .> threshold]
        newtgt = newtgt[:, E .> threshold]
    end

    src, newtgt
end
