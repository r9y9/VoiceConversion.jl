D = 20
T = 30

W = VoiceConversion.constructW(D, T)

@test issparse(W)
@test size(W) == (2D*T, D*T)

for t=1:T
    # coef. for static feature
    s = 2D*(t-1)+1
    e = s + D - 1
    @test W[s:e,(t-1)*D+1:t*D] == speye(D)
    for i=1:T
        if i != t
            @test W[s:e, (i-1)*D+1:i*D] == spzeros(D, D)
        end
    end
    
    # coef. for dynamic feature
    s = s + D
    e = e + D
    if t >= 2
        @test W[s:e, (t-2)*D+1:(t-1)*D] == -0.5*speye(D)
    end
    if t < T
        @test W[s:e, t*D+1:(t+1)*D] == 0.5*speye(D)
    end
    
    for i=1:T
        if i != t-1 && i != t+1
             @test W[s:e, (i-1)*D+1:i*D] == spzeros(D, D)
        end
    end
end
