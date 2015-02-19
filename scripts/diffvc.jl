using DocOpt

doc="""Voice conversion based on differencial spectral compensation

Usage:
    diffvc.jl [options] <input_wav> <model_jld> <dst_wav>
    diffvc.jl --version
    diffvc.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 40]
    --alpha=ALPHA    all-pass constant [default: 0.0]
    --trajectory     trajectory-based parameter conversion
    --T=t            maximum length for one-step trajectory conversion [default: 100]
"""

using VoiceConversion
using MelGeneralizedCepstrums
using WAV
using SynthesisFilters
using HDF5, JLD
using WORLD

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_wav>"])
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float(vec(x))
    fs = int(fs)
    @info("length of input signal is $(length(x)/fs) sec.")

    period = float(args["--period"])
    order = int(args["--order"])
    α = float(args["--alpha"])
    if α == 0.0
        α = mcepalpha(fs)
    end
    trajectory = args["--trajectory"]

    # Load mapping model
    gmm = load(args["<model_jld>"])
    if !gmm["diff"]
        @warn("The model doesn't seem to be trained on differencial features")
    end

    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    if trajectory
        mapper = TrajectoryGMMMap(mapper, int(args["--T"]))
    end

    # shape (order+1, number of frames)
    elapsed_fe = @elapsed begin
        w = World(fs, period)
        f0, timeaxis = dio(w, x)
        f0 = stonemask(w, x, timeaxis, f0)
        spectrogram = cheaptrick(w, x, timeaxis, f0)
        src = sp2mc(spectrogram, order, α)
    end
    @info("elapsed time in feature extraction is $(elapsed_fe) sec.")
    if trajectory
        # add delta feature
        src = [src[1,:], push_delta(src[2:end,:])]
    end

    # remove power coef.
    src[1,:] = 0.0

    # Perform conversion
    elapsed_vc = @elapsed converted = vc(mapper, src)
    @info("elapsed time in conversion process is $(elapsed_vc) sec.")


    # Waveform synthesis using Mel-Log Spectrum Approximation filter
    mf = MLSADF(order, α)
    hopsize = int(fs / (1000 / period))

    elapsed_syn = @elapsed y = synthesis!(mf, x, mc2b(converted, α), hopsize)
    @info("elapsed time in waveform moduration is $(elapsed_syn) sec.")

    wavwrite(float(y), args["<dst_wav>"], Fs=fs)
    @info("Dumped to ", args["<dst_wav>"])
end

main()
