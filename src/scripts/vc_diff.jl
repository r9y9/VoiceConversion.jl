using DocOpt

doc="""Voice conversion based on differencial spectral compensation

Usage:
    vc_diff.jl [options] <input_wav> <model_jld> <dst_wav>
    vc_diff.jl --version
    vc_diff.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 25]
    --alpha=ALPHA    all-pass constant [default: 0.0]
    --trajectory     trajectory-based parameter conversion
"""

using VoiceConversion
using MCepAlpha
using WAV
using SPTK
using HDF5, JLD

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_wav>"], format="int")
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float64(x[:])
    const fs = int(fs)
    println("length of input signal is $(length(x)/fs) sec.")
    
    const period = float(args["--period"])
    const order = int(args["--order"])
    alpha = float(args["--alpha"])
    if alpha == 0.0
        alpha = mcepalpha(fs)
    end
    const trajectory = args["--trajectory"]
        
    # Load mapping model
    gmm = load(args["<model_jld>"])
    if !gmm["diff"]
        error("not supported")
    end
    mapper = GMMMap(gmm)
    if trajectory
        mapper = TrajectoryGMMMap(mapper, 50)
    end

    # shape (order+1, number of frames)
    elapsed_fe = @elapsed src = world_mcep(x, fs, period, order, alpha)
    println("elapsed time in feature extraction is $(elapsed_fe) sec.")
    if trajectory
        # add delta feature
        src = [src[1,:], push_delta(src[2:end,:])]
    end

    # remove power coef.
    src[1,:] = 0.0
    
    # Perform conversion
    elapsed_vc = @elapsed converted = vc(mapper, src)
    println("elapsed time in conversion process is $(elapsed_vc) sec.")
    
    
    # Waveform synthesis using Mel-Log Spectrum Approximation filter
    mf = MLSADF(order)
    hopsize = int(fs / (1000 / period))
    elapsed_syn = @elapsed begin
        synthesized = synthesis!(mf, x, converted, alpha, hopsize)
    end
    println("elapsed time in waveform moduration is $(elapsed_syn) sec.")

    wavwrite(int16(synthesized), args["<dst_wav>"], Fs=fs)
    println("Dumped to ", args["<dst_wav>"])
end

@time main()
