# WORLD-based f0 estimation
function wf0(x::AbstractVector,
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

    f0, timeaxis
end

function save_wf0(savepath,
                  f0::AbstractVector,
                  fs::Integer,
                  period::FloatingPoint,
                  )
    save(savepath,
         "description", "WORLD-based F0",
         "type", "f0",
         "fs", fs,
         "period", period,
         "feature_vector", f0
         )
end
