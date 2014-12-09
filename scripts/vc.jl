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
using MCepAlpha
using WAV
using HDF5, JLD
using WORLD

function main()
    args = docopt(doc, version=v"0.0.1")

    x, fs = wavread(args["<input_wav>"])
    @assert size(x, 2) == 1 "The input data must be monoral."
    x = float(vec(x))
    const fs = int(fs)
    println("length of input signal is $(length(x)/fs) sec.")
    
    const period = float(args["--period"])
    const order = int(args["--order"])
    alpha = float(args["--alpha"])
    if alpha == 0.0
        alpha = mcepalpha(fs)
    end
    const trajectory = args["--trajectory"]
    const gvmodel = string(args["--gv"])
        
    # Load mapping model
    gmm = load(args["<model_jld>"])
    if gmm["diff"]
        warn("The model seem to be trained on differencial features")
    end

    mapper = GMMMap(gmm["weights"], gmm["means"], gmm["covars"])
    if trajectory
        mapper = TrajectoryGMMMap(mapper, int(args["--T"]))
    end

    if trajectory && !isempty(gvmodel)
        gv = load(gvmodel)
        gv["n_components"] == 1 || error("only single Gaussian for GV is supported")
        μᵛ = gv["means"][:,1]
        Σᵛᵛ = gv["covars"][:,:,1]
        mapper = TrajectoryGVGMMMap(mapper, μᵛ, Σᵛᵛ)
    end

    elapsed_fe = @elapsed begin
        w = World(fs=fs, period=period)
        
        # Fundamental frequency (f0) estimation by DIO
        f0, timeaxis = dio(w, x)
        
        # F0 re-estimation by StoneMask
        f0 = stonemask(w, x, timeaxis, f0)
        
        # Spectral envelope estimation
        spectrogram = cheaptrick(w, x, timeaxis, f0)

        # Spectral envelop -> Mel-cesptrum
        src = wsp2mc(spectrogram, order, alpha)

        # aperiodicity ratio estimation
        ap = aperiodicityratio(w, x, f0, timeaxis)
    end

    println("elapsed time in feature extraction is $(elapsed_fe) sec.")
    if trajectory
        # add delta feature
        src = [src[1,:], push_delta(src[2:end,:])]
    end
    
    # Perform conversion
    elapsed_vc = @elapsed converted = vc(mapper, src)
    println("elapsed time in conversion process is $(elapsed_vc) sec.")
    
    # Mel-Cepstrum to spectrum
    converted_spectrogram = mc2wsp(converted, size(spectrogram,1), -alpha)

    # Waveform synthesis using WORLD
    elapsed_syn = @elapsed begin
        y = synthesis_from_aperiodicity(w, f0, converted_spectrogram, ap,
                                        length(x))
    end
    println("elapsed time in waveform synthesis is $(elapsed_syn) sec.")
    
    wavwrite(float(y), args["<dst_wav>"], Fs=fs)
    println("Dumped to ", args["<dst_wav>"])
end

@time main()
