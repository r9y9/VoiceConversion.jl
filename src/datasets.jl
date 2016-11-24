# Dataset types and utility functions to construct datasets

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

# TODO(ryuichi): should be more generic
function push_delta(src)
    D, T = size(src)
    src = repmat(src, 2)
    for t=2:T-1
        src[D+1:end,t] = -0.5*src[1:D,t-1] + 0.5*src[1:D,t+1]
    end
    src
end

abstract Dataset

immutable ParallelDataset <: Dataset
    X::Matrix{Float64}
    Y::Matrix{Float64}

    # keep statistics in case of recovering original data from
    # standarized data
    Xmean::Matrix{Float64}
    Xstd::Matrix{Float64}
    Ymean::Matrix{Float64}
    Ystd::Matrix{Float64}

    diff::Bool

    function ParallelDataset(path;
                             diff::Bool=false,
                             joint::Bool=true,
                             standarize::Bool=false,
                             ignore0th::Bool=true,
                             add_delta::Bool=false,
                             suffix::AbstractString="_parallel.jld",
                             keepstat::Bool=false,
                             nmax::Int=100)
        files = searchdir(path, suffix)
        sort!(files)

        println("$(length(files)) training data found")

        XY = ones(1,1)
        totalframes::Int = 0
        totalphrases::Int = 0

        count = 0
        for filename in files
            # TODO(ryuichi) allow costom file format?
            f = load(joinpath(path, filename))
            src, tgt = f["src"], f["tgt"]
            src_x, tgt_x = src["feature_matrix"], tgt["feature_matrix"]

            if ignore0th
                src_x, tgt_x = src_x[2:end,:], tgt_x[2:end,:]
            end

            if add_delta
                src_x = push_delta(src_x)
                tgt_x = push_delta(tgt_x)
            end

            # use differencial
            if diff
                tgt_x = tgt_x - src_x
            end

            # Create joint features matries of source and target speaker
            combined = vcat(src_x, tgt_x)
            @assert size(combined) == (size(src_x, 1)*2, size(src_x, 2))

            if totalframes == 0
                XY = combined
            else
                XY = hcat(XY, combined)
            end

            totalframes += size(combined, 2)
            totalphrases += 1

            count += 1
            count >= nmax && break
        end

        println("total number of frames: $(totalframes)")
        println("total number of phrases: $(totalphrases)")

        X = ones(1,1)
        Y = ones(1,1)
        if joint
            X = XY
            Y = ones(1,1) # not used
        else
            D = div(size(XY, 1), 2)
            X = XY[1:D,:]
            Y = XY[D+1:end,:]
        end

        Xmean = mean(X,2)
        Xstd = std(X,2)
        Ymean = mean(Y,2)
        Ystd = std(Y,2)
        if standarize
            X = (X - Xmean) / Xstd
            Y = (Y - Ymean) / Ystd
        end

        # avoid nan
        if joint
            Ymean = ones(1,1)
            Ystd = ones(1,1)
        end

        @assert all(isfinite.(X))
        @assert all(isfinite.(Y))
        @assert all(isfinite.(Xmean))
        @assert all(isfinite.(Xstd))
        @assert all(isfinite.(Ymean))
        @assert all(isfinite.(Ystd))

        if !keepstat
            Xmean = ones(1,1)
            Xstd = ones(1,1)
            Ymean = ones(1,1)
            Ystd = ones(1,1)
        end

        new(X, Y, Xmean, Xstd, Ymean, Ystd, diff)
    end
end

# GVDataset represents a global variance dataset.
immutable GVDataset <: Dataset
    X::Matrix{Float64}

    function GVDataset(path;
                       ignore0th::Bool=true,
                       add_delta::Bool=false,
                       suffix::AbstractString=".jld",
                       nmax::Int=100)
        files = searchdir(path, suffix)
        sort!(files)

        println("$(length(files)) training data found")

        X = zeros(0, 0)
        totalphrases::Int = 0

        for filename in files
            # TODO(ryuichi) allow costom file format?
            f = load(joinpath(path, filename))
            tgt = f["feature_matrix"]

            if ignore0th
                tgt = tgt[2:end,:]
            end

            if add_delta
                tgt = push_delta(tgt)
            end

            gv = var(tgt, 2)
            if sum(isnan.(gv)) == 0
                if totalphrases == 0
                    X = gv
                else
                    X = hcat(X, gv)
                end
            end

            totalphrases += 1

            totalphrases >= nmax && break
        end

        println("total number of phrases: $(totalphrases)")

        @assert all(isfinite.(X))

        new(X)
    end
end
