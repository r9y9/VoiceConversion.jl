# TODO(ryuichi) should be more generic
# type DTW represents a dynamic time warping.
type DTW
    fstep::Int  # forward transition constraint
    bstep::Int  # backward transition constraint
    template::Matrix{Float64}
    costtable::Matrix{Float64}
    backpointer::Matrix{Int}
end

# constructor
function DTW(;fstep=0, bstep=1)
    DTW(fstep, bstep, zeros(1,1), zeros(1,1), zeros(Int,1,1))
end

function transition(d::DTW, i::Int, j::Int)
    if j == i+1
        return 0
    elseif i == j
        return 1
    else
        return 2
    end
end

function observation(d::DTW, v::Vector{Float64}, i::Int)
    return sumabs2(v  - d.template[:,i])
end

# Init state initialization
function lazy_init!(d::DTW, T::Int)
    d.costtable = reshape([1:T], T, 1)
    d.backpointer = reshape([1:T], T, 1)
end

function set_template!(d::DTW, template::Matrix{Float64})
    d.template = template
    lazy_init!(d, size(template, 2))
end

# online interface (slow)
function update!(d::DTW, v::Vector{Float64})
    # N is the dimention of a template, T is the current time length
    N, T = size(d.costtable)
    lastcost = d.costtable[:, T]
    currentcost = zeros(N)
    current_backpointer = zeros(Int, N)

    for i=1:N
        minindex = i
        const o, t = observation(d, v, i), transition(d, minindex, i)
        mincost = lastcost[minindex] + o + t

        # search minindex
        for j=i-d.bstep:i+d.fstep
            if j < 1 || j > N
                continue
            end
            cost = lastcost[j] + o + transition(d, j, i)
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

function fit!(d::DTW, template::Matrix, sequence::Matrix{Float64})
    set_template!(d, template)
    fit!(d, sequence)
end

function fit!(d::DTW, sequence::Matrix{Float64})
    for i=1:size(sequence, 2)
        update!(d, sequence[:,i])
    end

    return backward(d)
end

# backward searches the path that minimizes the total cost.
function backward(d::DTW)
    T  = size(d.costtable, 2) - 1 # exclude the initial state
    minpath = zeros(Int, T)

    minpath[end] = indmin(d.costtable[:,T+1])

    # revursive search
    for i=reverse(2:T)
        minpath[i-1] = d.backpointer[minpath[i], i+1]
    end

    return minpath
end
