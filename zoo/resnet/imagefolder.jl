using Images
using ImageView


function float2img(a::AbstractArray{T, 3}) where {T <: Real}
    img = permutedims(a, (3, 1, 2))
    img = colorview(RGB, img)
    return img
end

function img2float(img::Array{RGB{T}}) where T
    a = channelview(img)
    a = rawview(a)
    a = permutedims(a, (2,3,1))
    a = a / 255
    return a
end

img2float(img::Array{Gray{T}}) where T = img2float(RGB.(img))


function load_image(path::AbstractString, transform=nothing)
    img = load(path)
    if transform != nothing
        img = transform(img)
    end
    return img2float(img)
end


function view_image(a::AbstractArray{<:Real, 3})
    img = float2img(a)
    imshow(img)
end


struct ImageFolder
    dirpath::AbstractString
    paths::Vector{String}
    labels::Vector{Int}
    classes::Vector{String}
    transform::Function
end


function ImageFolder(dirpath::AbstractString; transform::Function=identity)
    paths = String[]
    classes = String[]
    labels = Int[]
    for subdir in readdir(dirpath)
        subdir_path = joinpath(dirpath, subdir)
        for filename in readdir(subdir_path)
            path = joinpath(subdir_path, filename)
            push!(paths, path)
            if isempty(classes) || classes[end] != subdir
                push!(classes, subdir)
            end
            push!(labels, length(classes))
        end
    end
    return ImageFolder(dirpath, paths, labels, classes, transform)
end

Base.show(io::IO, dataset::ImageFolder) = print(io, "ImageFolder(\"$(dataset.dirpath)\")")
Base.length(dataset::ImageFolder) = length(dataset.labels)


function Base.getindex(dataset::ImageFolder, idxs)
    labels = dataset.labels[idxs]
    imgs = Vector{Array}(undef, length(idxs))
    # Threads.@threads - causes concurrency violation exception from FileIO
    for (i, idx) in collect(enumerate(idxs))
        img = load(dataset.paths[idx])
        img = dataset.transform(img)
        arr = img2float(img)
        imgs[i] = arr
    end
    @assert(all(sz -> sz == size(imgs[1]), [size(img) for img in imgs]),
            "Cannot concat images into a batch since they have different size. " *
            "Use `transform` keyword argument to ImageFolder constructor to resize " *
            "all images to the same size.")
    imgs4d = cat([reshape(img, size(img)..., 1) for img in imgs]...; dims=4)
    return imgs4d, labels
end


function Lilith.getbatch(dataset::ImageFolder, i::Int, sz::Int)
    start = (i-1)*sz + 1
    finish = min(i*sz, length(dataset))
    start > finish && return nothing
    return dataset[start:finish]
end
