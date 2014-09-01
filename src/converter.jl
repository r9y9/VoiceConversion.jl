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

function vc(c::TrajectoryConverter, fm::Matrix{Float64})
    # TODO
end
