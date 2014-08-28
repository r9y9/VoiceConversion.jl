import SPTK
const sptk = SPTK

# logamp2mcep converts log-amplitude spectrum to mel-cepstrum.
function logamp2mcep(logamp::Vector{Float64}, order::Int, alpha::Float64)
    ceps = real(ifft(logamp))
    ceps[1] /= 2.0
    return sptk.freqt(ceps, order, alpha)
end

# mcep2e computes energy from mel-cepstrum.
function mcep2e(mc::Vector{Float64}, alpha::Float64, len::Int)
    # back to linear frequency domain
    c = sptk.freqt(mc, len-1, -alpha)

    # compute impule response from cepsturm
    ir = sptk.c2ir(c, len)

    return sumabs2(ir)
end

mcep2e(mat::Matrix{Float64}, alpha, len) =
    [mcep2e(mat[:,i], alpha, len) for i=1:size(mat, 2)]
