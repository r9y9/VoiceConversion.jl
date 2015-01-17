# Alignment functions to create parallel data

# general feature alignment
function _align(src::AbstractMatrix, # source feature matrix
                tgt::AbstractMatrix  # target feature matrix
                )
    if size(src, 1) != size(tgt, 1)
        throw(DimentionMismatch("order of feature vector must be equal"))
    end

    # Alignment
    d = DTW(fstep=0, bstep=2) # allow one skip
    path = fit!(d, src, tgt)

    # create aligned tgt
    newtgt = zeros(eltype(src), size(src))
    newtgt[:,path] = tgt[:,1:length(path)]

    # interpolation
    # TODO(ryuichi) better solution
    hole = setdiff([path[1]:path[end]], path)
    for i in hole
        if i > 1 && i < size(src, 2)
            for j=1:size(newtgt, 1)
                @inbounds newtgt[j,i] = (newtgt[j,i-1] + newtgt[j,i+1]) / 2.0
            end
        end
    end

    src, newtgt
end

# alignment function specialized for mel-cepstrum
function _align_mcep(src::AbstractMatrix,       # source feature matrix
                     tgt::AbstractMatrix,       # target feature matrix
                     α::FloatingPoint,         # all-pass constant
                     fftlen::Integer;           # fft length
                     threshold::Float64=-14.0,  # threshold to remove silence frames
                     remove_silence::Bool=true
                     )
    src, newtgt = align(src, tgt)

    if remove_silence
        E = log(mc2e(src, α, fftlen))
        src = src[:, E .> threshold]
        newtgt = newtgt[:, E .> threshold]
    end

    src, newtgt
end

function align_save(src, tgt, dst)
    save(dstpath,
         "description", "Parallel data",
         "src", src,
         "tgt", tgt
         )
end

function align(srcpath,
               tgtpath,
               dstpath;
               threshold::FloatingPoint=-14.0,
               )
    src = load(srcpath)
    tgt = load(tgtpath)

    src_fm = src["feature_matrix"]
    tgt_fm = tgt["feature_matrix"]

    src_fm, tgt_fm = zeros(0,0), zeros(0,0)
    if src["type"] == "MelCepstrum"
        src_fm, tgt_fm = _align_mcep(src_fm, tgt_fm,
                                     float(src["alpha"]),
                                     int(src["fftlen"]);
                                     threshold=threshold)
    else
        src_fm, tgt_fm = _align(src_fm, tgt_fm)
    end

    @assert size(src_fm) == size(tgt_fm)

    @info("The number of aligned frames: $(size(src_fm, 2))")
    if size(src_fm, 2) ==  0
        @warn("No frame found in aligned data. Probably threshold is too high.")
    end

    @assert !any(isnan(src_fm))
    @assert !any(isnan(tgt_fm))

    src["feature_matrix"] = src_fm
    tgt["feature_matrix"] = tgt_fm

    # type Dict{Union(UTF8String, ASCIIString), Any} is saved as
    # Dict{UTF8String, Any} and cause error in reading JLD file.
    # remove off Union and then save do the trick (but why? bug in HDF5?)
    @assert isa(src, Dict{Union(UTF8String, ASCIIString), Any})
    @assert isa(tgt, Dict{Union(UTF8String, ASCIIString), Any})

    align_save(Dict{UTF8String, Any}(src),
               Dict{UTF8String, Any}(tgt),
               dstpath)
end
