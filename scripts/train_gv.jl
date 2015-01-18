using DocOpt

doc="""Training (Single) Gaussian Mixture model for global variance.

Usage:
    train_gv.jl [options] <tgt_dir> <dst_jld>
    train_gv.jl --version
    train_gv.jl -h | --help

Options:
    -h --help           show this message
    --add_delta         add delta feature
    --n_components=MIX  number of mixtures [default: 1]
    --n_iter=ITER       number of iterations [default: 200]
    --n_init=N          number of initialization [default: 2]
    --max=MAX           maximum number of data we use [default: 100]
    --min_covar=MIN     minimum covariance [default: 1.0e-7]
"""

using VoiceConversion
using VoiceConversion.Tools

let
    args = docopt(doc, version=v"0.0.1")

    dataset = GVDataset(args["<tgt_dir>"],
                        add_delta=args["--add_delta"],
                        nmax=int(args["--max"]))

    savepath = args["<dst_jld>"]

    gmm = train_gmm(dataset, savepath;
                    n_components=int(args["--n_components"]),
                    n_iter=int(args["--n_iter"]),
                    n_init=int(args["--n_init"]),
                    min_covar=float64(args["--min_covar"]),
                    refine=false
                    )

    save_gmm(savepath, gmm)

    println("Finished")
end
