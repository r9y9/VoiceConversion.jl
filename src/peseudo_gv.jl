# Peseudo GV emphasis filter
type PeseudoGV
    p::Float64
end

const magic_paramter = 1.4 
PeseudoGV(;p::Float64=magic_paramter) = PeseudoGV(p)

function fvpostf!(pgv::PeseudoGV, src::AbstractVector)
    # simply multiply constant
    src *= pgv.p
    nothing
end

function fvpostf(pgv::PeseudoGV, src::AbstractVector)
    filtered = copy(src)
    fvpostf!(pgv, filtered)
    return filtered
end
