## generic rrule for broadcasted that we DON'T use can be found here:
## https://github.com/JuliaDiff/ChainRules.jl/issues/531
## below are implementations of rrule for broadcasting of the most popular functions


function rrule(::typeof(Broadcast.broadcasted), ::typeof(tanh), x)
  y = tanh.(x)
  function bcast_tanh_pullback(dy)
    dx = @. (1 - y ^ 2) * dy
    return NoTangent(), NoTangent(), dx
  end
  return y, bcast_tanh_pullback
end



