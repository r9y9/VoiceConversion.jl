module DTWs

## A small module to perform Dynamic Time Warping alignment. ##

export DTW, fit!, update!, set_template!, backward

# type DTW represents a dynamic time warping.
# TODO(ryuichi) should be more generic
type DTW
    fstep::Int  # forward transition constraint
    bstep::Int  # backward transition constraint
    template::Matrix{Float64}
    costtable::Matrix{Float64}
    backpointer::Matrix{Int}
end

function DTW(;fstep=0, bstep=1)
    DTW(fstep, bstep, zeros(1, 1), zeros(1, 1), zeros(Int, 1, 1))
end

function transition(d::DTW, i::Int, j::Int)
    if j == i+1
        return 0.0
    elseif i == j
        return 1.0
    end

    2.0 # abs(i-j)
end

function observation(d::DTW, v::AbstractVector, i::Int)
    sumabs2(v  - d.template[:,i])
end

# lazy_init! performs state initialization.
function lazy_init!(d::DTW, S::Int)
    d.costtable = reshape(1:S, S, 1)
    d.backpointer = reshape(1:S, S, 1)
end

# lazy_init! performs pre-allocations
function lazy_init!(d::DTW, S::Int, T::Int)
    # plus initial state
    d.costtable = zeros(Float64, S, T+1)
    d.backpointer = ones(Int, S, T+1)

    d.costtable[:,1] = [1:S;]
    d.backpointer[:,1] = [1:S;]
end

function set_template!(d::DTW, template::Matrix{Float64})
    d.template = template
    lazy_init!(d, size(template, 2))
end

# update! updates cost table for a given vector in an on-line manner.
# In off-line situations, please use fit! because it is more efficient
# and faster than this on-line version.
function update!(d::DTW, v::AbstractVector)
    # S: length of tempalte, T: current time length
    const S, T = size(d.costtable)
    const lastcost = d.costtable[:, T]
    currentcost = zeros(S)
    current_backpointer = zeros(Int, S)

    for i=1:S
        minindex = i
        const obs = observation(d, v, i)
        const trans = transition(d, minindex, i)
        mincost = lastcost[minindex] + obs + trans

        # search minindex
        for j=i-d.bstep:i+d.fstep
            if j < 1 || j > S
                continue
            end
            cost = lastcost[j] + obs + transition(d, j, i)
            if cost < mincost
                mincost, minindex = cost, j
            end
        end
        currentcost[i] = mincost
        current_backpointer[i] = minindex
    end

    d.costtable = [d.costtable currentcost]
    d.backpointer = [d.backpointer current_backpointer]
end

# fit! aligns two sequences using dynamic time warping algorithm.
function fit!(d::DTW, template::Matrix{Float64}, sequence::Matrix{Float64})
    # S: length of template, T: length of target sequence
    const S, T = size(template, 2), size(sequence, 2)

    # pre-allocations
    lazy_init!(d, S, T)
    @assert size(d.costtable) == (S, T+1)

    # set template matrix to align
    d.template = template

    for t=1:T
        v = sequence[:,t]
        for i=1:S
            minindex = i
            ocost = observation(d, v, i)
            tcost = transition(d, minindex, i)
            @inbounds mincost = d.costtable[minindex, t] + ocost + tcost

            # search minindex
            for j=i-d.bstep:i+d.fstep
                if j < 1 || j > S
                    continue
                end
                @inbounds cost = d.costtable[j, t] + ocost + transition(d, j, i)
                if cost < mincost
                    mincost, minindex = cost, j
                end
            end
            @inbounds d.costtable[i, t+1] = mincost
            @inbounds d.backpointer[i, t+1] = minindex
        end
    end

    backward(d)
end

fit!(d::DTW, sequence::Matrix{Float64}) = fit!(d, d.template, sequence)

# backward searches the path that minimizes the total cost.
function backward(d::DTW)
    const T  = size(d.costtable, 2) - 1 # exclude the initial state
    minpath = zeros(Int, T)

    minpath[end] = indmin(d.costtable[:,T+1])

    # revursive search
    for i=reverse(2:T)
        @inbounds minpath[i-1] = d.backpointer[minpath[i], i+1]
    end

    minpath
end

end # module
