using DocOpt

doc="""Voice conversion based on WORLD-based speech analysis and
synthesis framework

Usage:
    vc.jl [options] <input_wav> <model_jld> <dst_wav>
    vc.jl --version
    vc.jl -h | --help

Options:
    -h --help        show this message
    --period=PERIOD  frame period in msec [default: 5.0]
    --order=ORDER    order of mel cepsrum [default: 25]
    --alpha=ALPHA    all-pass constant [default: 0.0]
    --gv=MODEL       global variance [default: ]
    --trajectory     trajectory-based parameter conversion
    --T=t            maximum length for one-step trajectory conversion [default: 100]
"""

using VoiceConversion
using MelGeneralizedCepstrums
using WAV
using HDF5, JLD
using WORLD

using Logging
@Logging.configure(level=DEBUG, output=STDOUT)

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_wav>"])
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = map(Float64, vec(x))
    fs = Int(fs)
    @info("length of input signal is $(length(x)/fs) sec.")

    period = parse(Float64, args["--period"])
    order = parse(Int, args["--order"])
    α = parse(Float64, args["--alpha"])
    if α == 0.0
        α = mcepalpha(fs)
    end
    trajectory = args["--trajectory"]
    gvmodel = string(args["--gv"])

    # Load mapping model
    gmm = load(args["<model_jld>"])
    if gmm["diff"]
        @warn("The model seem to be trained on differencial features")
    end

    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    if trajectory
        mapper = TrajectoryGMMMap(mapper, parse(Int, args["--T"]))
    end

    if trajectory && !isempty(gvmodel)
        gv = load(gvmodel)
        gv["n_components"] == 1 || error("only single Gaussian for GV is supported")
        μᵛ = gv["means"][:,1]
        Σᵛᵛ = gv["covars"][:,:,1]
        mapper = TrajectoryGVGMMMap(mapper, μᵛ, Σᵛᵛ)
    end

    elapsed_fe = @elapsed begin
        f0, timeaxis = dio(x, fs, DioOption(period=period))
        f0 = stonemask(x, fs, timeaxis, f0)
        spectrogram = cheaptrick(x, fs, timeaxis, f0)
        src = sp2mc(spectrogram, order, α)
        ap = d4c(x, fs, timeaxis, f0)
    end

    @info("elapsed time in feature extraction is $(elapsed_fe) sec.")
    if trajectory
        # add delta feature
        src = [reshape(src[1,:], 1, size(src,2)); push_delta(src[2:end,:])]
    end

    # Perform conversion
    elapsed_vc = @elapsed converted = vc(mapper, src)
    @info("elapsed time in conversion process is $(elapsed_vc) sec.")

    # Mel-Cepstrum to spectrum
    fftlen = size(spectrogram,1)*2-1
    converted_spectrogram = mc2sp(converted, α, fftlen)

    # Waveform synthesis using WORLD
    elapsed_syn = @elapsed begin
        y = synthesis(f0, converted_spectrogram, ap, period, fs, length(x))
    end
    @info("elapsed time in waveform synthesis is $(elapsed_syn) sec.")

    wavwrite(map(Float64, y), args["<dst_wav>"], Fs=fs)
    @info("Dumped to ", args["<dst_wav>"])
end

main()
