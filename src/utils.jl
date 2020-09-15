
function norm2_values(x)
    if x isa Dict
        vals = collect(values(x))
        return vals
    elseif x isa AbstractArray || x isa Tuple
        return flatten(map(norm2_values, x))
    elseif x isa Function
        return []
    elseif Yota.isstruct(x)
        sub_iters = []
        for fld in fieldnames(typeof(x))
            push!(sub_iters, norm2_values(getfield(x, fld)))
        end
        return flatten(sub_iters)
    else
        return x
    end
end


function norm2(x)
    Statistics.norm(collect(norm2_values(x)), 2)
end


"""
Walk fields of struct `s` applying callback function `cb`.
If `cb` returns nothing, recursively call `walkstruct()` on each field.
"""
function walkstruct(cb::Function, s)
    rec = cb(s)
    if rec == nothing
        for p_name in propertynames(s)
            prop = getproperty(s, p_name)
            walkstruct(cb, prop)
        end
    end
end
