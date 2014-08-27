import SPTK
const sptk = SPTK

# logamp2mcep converts log-amplitude spectrum to mel-cepstrum.
function logamp2mcep(logamp::Vector{Float64}, order::Int, alpha::Float64)
    ceps = real(ifft(logamp))
    ceps[1] /= 2.0
    return sptk.freqt(ceps, order, alpha)
end
