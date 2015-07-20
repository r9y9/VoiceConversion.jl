## Type Hierarchy ##
abstract AbstractConverter
abstract FrameByFrameConverter <: AbstractConverter
abstract TrajectoryConverter <: AbstractConverter

# vc performs voice conversion based on specified converter.
function vc(c::FrameByFrameConverter,
            fm::AbstractMatrix{Float64} # feature matrix
            )
    # Split src feature matrix to power and spectral features
    power, src =  fm[1,:], fm[2:end,:]

    T = size(src, 2)
    converted = similar(fm)

    # Perform feature mapping for each time
    for t=1:T
        @inbounds converted[2:end,t] = fvconvert(c, src[:,t])
    end

    # keep original power
    # note that it is assumed that 0-th coef. represents power coef.
    converted[1,:] = power

    converted
end

# vc performs trajectory-based feature conversion. To reduce computational
# complexiy in practice, we split the input sequence to a set of sub-sequences
# and perform trajectory-based conversion for each sub-sequence.
function vc(c::TrajectoryConverter,
            fm::AbstractMatrix{Float64};  # feature matrix
            )
    # Split src feature matrix to power and spectral features
    power, src =  fm[1,:], fm[2:end,:]

    # D = length/2 + 1: order of static spectral features + power
    D, T = size(src, 1)>>1 + 1, size(src, 2)
    converted = Array(eltype(fm), D, T)

    # Perform Trajectory-based mapping
    count = 0
    limit = length(c) # maximum length of sub-sequence
    while true
        b = count * limit + 1
        e = (count+1) * limit
        if e > T
            e = T
        end
        phrase = src[:,b:e]
        converted[2:end,b:e] = fvconvert(c, phrase)

        if e == T
            break
        end
        count += 1
    end

    # keep original power
    converted[1,:] = power

    converted
end
