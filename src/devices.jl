abstract type AbstractDevice end

struct CPU <: AbstractDevice
end

struct GPU <: AbstractDevice
    id::Int
end

GPU() = GPU(1)


to_device(device::CPU, x::CuArray) = convert(Array, x)


function to_device(device::GPU, x)
    T = typeof(x)
    flds = fieldnames(T)
    if isa(x, CuArray)
        return x
    elseif isa(x, AbstractFloat)
        return Float32(x)
    elseif isa(x, Tuple)
        return ((to_device(device, el) for el in x)...,)
    elseif isempty(flds)
        # primitive or array
        return cu(x)
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_device(device, getfield(x, fld)) for fld in flds]
        return T(fld_vals...)
    end
end

# is_cuarray(x) = x isa CuArray


## currently GPU's ID is just a placeholder
# guess_device(args) = any(is_cuarray, args) ? GPU(1) : CPU()
# device_of(A) = A isa CuArray ? GPU(1) : CPU()


# """
# Retrieve function compatible with specified device
# See also: to_device(device, f)
# """
# device_function(device::CPU, f) = f


"""
Convert object to a compatible with the specified device.
For CPU it's usually no-op. For GPU behavior differs between object types:
 * Arrays are converted to CuArrays
 * structs are converted recursively
 * functions are looked up using `device_function()` or transformed using tracer
 * all other objects are returned as is
"""
to_device(device::CPU, x) = x
to_device(device::CPU, f::Function, args) = f


(device::AbstractDevice)(x) = to_device(device, x)


# to_same_device(A, example) = device_of(example)(A)