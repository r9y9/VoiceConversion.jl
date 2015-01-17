# WORLD-based f0 estimation
function _wf0(x::AbstractVector,
              fs::Integer,
              period::FloatingPoint;
              opt::DioOption=WORLD.defaultdioopt,
              f0refine::Bool=true,
              )
    # F0 estimation
    w = World(fs, period)

    f0, timeaxis = dio(w, x; opt=opt)

    # F0 re-estimation by StoneMask
    if f0refine
        f0 = stonemask(w, x, timeaxis, f0)
    end

    f0
end

function wf0_save(f0::AbstractVector,
                  fs::Integer,
                  period::FloatingPoint,
                  dstpath)
    save(dstpath,
         "description", "WORLD-based F0",
         "type", "f0",
         "fs", fs,
         "period", period,
         "feature_vector", f0
         )
end

function wf0(wavpath,
             period::FloatingPoint,
             dstpath;
             opt::DioOption=WORLD.defaultdioopt,
             f0refine::Bool=true,
             )
    x, fs = wavread(wavpath)
    size(x, 2) != 1 && error("The input data must be monoral.")
    x = vec(x)

    f0 = _wf0(x, fs, period, opt=opt, f0refine=f0refine)
    wf0_save(f0, fs, period, dstpath)
end
