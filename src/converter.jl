# vc performs voice conversion based on specified converter.
function vc(c::FrameByFrameConverter, fm::Matrix{Float64})
    # Split src feature matrix to power and spectral features
    power, src =  fm[1,:], fm[2:end,:]

    const D, T = size(src)
    converted = Array(eltype(fm), size(fm))

    # Perform feature mapping for each time
    for t=1:T
        converted[2:end,t] = fvconvert(c, src[:,t])
    end

    # keep power
    converted[1,:] = power

    return converted
end

function vc(c::TrajectoryConverter, fm::Matrix{Float64};
            limit::Int=70)
    # Split src feature matrix to power and spectral features
    power, src =  fm[1,:], fm[2:end,:]

    const D, T = size(src)
    # D/2 + 1: order of static spectral features + power
    converted = Array(eltype(fm), div(D,2)+1, T)

    # Perform Trajectory-based mapping
    # Split whole sequence to a set of phrases to reduce memory size
    # that is used in conversion process.
    # Conversion is performed for each phrase.
    count::Int = 0
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

    # keep power
    converted[1,:] = power

    return converted
end
