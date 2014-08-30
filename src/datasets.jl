using HDF5, JLD

abstract Dataset

searchdir(path, key) = filter(x -> contains(x, key), readdir(path))

immutable ParallelDataset <: Dataset
    X::Matrix{Float64}
    Y::Matrix{Float64}

    # keep statistics in case of recovering original data from
    # standarized data
    Xmean::Matrix{Float64}
    Xstd::Matrix{Float64}
    Ymean::Matrix{Float64}
    Ystd::Matrix{Float64}

    function ParallelDataset(path;
                             start=None,
                             stop=None,
                             diff::Bool=false,
                             joint::Bool=true,
                             standarize::Bool=false,
                             ignore0th::Bool=true,
                             add_dynamic::Bool=false,
                             suffix::String="_parallel.jld",
                             keepstat::Bool=false)
        files = searchdir(path, suffix)
        sort!(files)

        info("$(length(files)) training data found")

        XY = ones(1,1)
        totalframes::Int = 0
        totalphrases::Int = 0
        for filename in files
            # TODO(ryuichi) allow costom file format?
            f = load(joinpath(path, filename))
            src, tgt = f["src"], f["tgt"]
            src_x, tgt_x = src["feature_matrix"], tgt["feature_matrix"]

            if ignore0th
                src_x, tgt_x = src_x[2:end,:], tgt_x[2:end,:]
            end

            if add_dynamic
                #TODO
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
        end

        info("total number of frames: $(totalframes)")
        info("total number of phrases: $(totalphrases)")

        X = ones(1,1)
        Y = ones(1,1)
        if joint
            X = XY
            Y = ones(1,1) # not used
        else
            order::Int = int(size(XY, 1)/2)
            X = XY[1:order,:]
            Y = XY[order+1:end,:]
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

        @assert !any(isnan(X))
        @assert !any(isnan(Y))
        @assert !any(isnan(Xmean))
        @assert !any(isnan(Xstd))
        @assert !any(isnan(Ymean))
        @assert !any(isnan(Ystd))

        if !keepstat
            Xmean = ones(1,1)
            Xstd = ones(1,1)
            Ymean = ones(1,1)
            Ystd = ones(1,1)
        end

        new(X, Y, Xmean, Xstd, Ymean, Ystd)
    end
end
