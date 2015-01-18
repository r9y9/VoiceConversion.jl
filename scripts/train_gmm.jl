using DocOpt

doc="""Training Gaussian Mixture models for Voice Conversion.

Usage:
    train_gmm.jl [options] <parallel_dir> <dst_jld>
    train_gmm.jl --version
    train_gmm.jl -h | --help

Options:
    -h --help           show this message
    --diff              use differencial
    --add_delta         add delta feature
    --n_components=MIX  number of mixtures [default: 16]
    --n_iter=ITER       number of iterations [default: 200]
    --n_init=N          number of initialization [default: 2]
    --max=MAX           maximum number of data we use [default: 100]
    --min_covar=MIN     minimum covariance [default: 1.0e-7]
    --refine            refine pre-trained GMM (<dst_jld>)
"""

using VoiceConversion
using VoiceConversion.Tools

let
    args = docopt(doc, version=v"0.0.2")

    dataset = ParallelDataset(args["<parallel_dir>"],
                              joint=true,
                              diff=args["--diff"],
                              add_delta=args["--add_delta"],
                              nmax=int(args["--max"]))

    savepath = args["<dst_jld>"]

    gmm = train_gmm(dataset, savepath;
                    n_components=int(args["--n_components"]),
                    n_iter=int(args["--n_iter"]),
                    n_init=int(args["--n_init"]),
                    min_covar=float64(args["--min_covar"]),
                    refine=args["--refine"],
                    pretrained_gmm_path=args["--refine"] ? savepath : nothing
                    )

    save_gmm(savepath, gmm, diff=dataset.diff)

    println("Finished")
end
