import Lilith.walkstruct


conv3x3(inplanes, outplanes; stride=1, dilation=1) =
    Conv2d(inplanes, outplanes, 3; stride=stride, padding=dilation, dilation=dilation, bias=false)

conv1x1(inplanes, outplanes; stride=1) =
    Conv2d(inplanes, outplanes, 1; stride=stride, bias=false)


################################################################################
#                                BasicBlock                                    #
################################################################################

mutable struct BasicBlock
    conv1::Conv2d       # 3x3
    bn1::BatchNorm2d
    conv2::Conv2d       # 3x3
    bn2::BatchNorm2d
    downsample::Union{Sequential, Nothing}
end


function BasicBlock(inplanes::Int, planes::Int; downsample=nothing, stride::Int=1,
                    base_width::Int=64, dilation=1)
    return BasicBlock(
        conv3x3(inplanes, planes; stride=stride),
        BatchNorm2d(planes),
        conv3x3(planes, planes),
        BatchNorm2d(planes),
        downsample
    )
end

expansion(m::Type{BasicBlock}) = 1


function (m::BasicBlock)(x::AbstractArray{T, 4}) where T
    identity = x
    out = m.conv1(x)
    out = m.bn1(out)
    out = relu.(out)
    out = m.conv2(out)
    out = m.bn2(out)
    if m.downsample != nothing
        identity = m.downsample(x)
    end
    out = out .+ identity
    out = relu.(out)
    return out
end


################################################################################
#                                Bottleneck                                    #
################################################################################


mutable struct Bottleneck
    conv1::Conv2d
    bn1::BatchNorm2d
    conv2::Conv2d
    bn2::BatchNorm2d
    conv3::Conv2d
    bn3::BatchNorm2d
    downsample::Union{Sequential, Nothing}
    stride
end

function Bottleneck(inplanes, planes; stride=1, downsample=nothing, base_width=64, dilation=1)
    width = Int(planes * (base_width / 64.0))
    expansion = 4
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    return Bottleneck(
        conv1x1(inplanes, width),
        BatchNorm2d(width),
        conv3x3(width, width; stride=stride, dilation=dilation),
        BatchNorm2d(width),
        conv1x1(width, planes * expansion),
        BatchNorm2d(planes * expansion),
        downsample,
        stride
    )
end

expansion(block::Type{Bottleneck}) = 4


function (m::Bottleneck)(x::AbstractArray{T, 4}) where T
    identity = x
    out = m.conv1(x)
    out = m.bn1(out)
    out = relu.(out)
    out = m.conv2(out)
    out = m.bn2(out)
    out = relu.(out)
    out = m.conv3(out)
    out = m.bn3(out)
    if m.downsample != nothing
        identity = m.downsample(x)
    end
    out = out .+ identity
    out = relu.(out)
    return out
end


################################################################################
#                                  ResNet                                      #
################################################################################


function make_layer(block, inplanes::Int, planes::Int, blocks::Int, base_width::Int; stride=1, dilation=1)
    norm_layer = BatchNorm2d
    # in the original code there's also `dilate` parameter which leads to pretty convoluted
    # stateful logic, thus we ignore it in the initial implementation
    # ResNet should still behave as usual given default parameters
    downsample = nothing
    if stride != 1 || inplanes != planes * expansion(block)
        downsample = Sequential(
            conv1x1(inplanes, planes * expansion(block); stride=stride),
            norm_layer(planes * expansion(block))
        )
    end
    layers = []
    push!(layers, block(inplanes, planes; stride=stride, downsample=downsample,
                        base_width=base_width, dilation=dilation))
    # expand input to the next set of blocks
    inplanes = planes * expansion(block)
    for _=2:blocks
        push!(layers, block(inplanes, planes; base_width=base_width, dilation=dilation))
    end
    return Sequential(layers...), inplanes
end


mutable struct ResNet
    conv1::Conv2d
    bn1::BatchNorm2d
    layer1::Sequential
    layer2::Sequential
    layer3::Sequential
    layer4::Sequential
    fc::Linear
end


function ResNet(block::Union{Type{BasicBlock}, Type{Bottleneck}}, block_nums::Vector{Int};
                num_classes=1000, zero_init_residual=false, width_per_group=64)
    norm_layer = BatchNorm2d
    inplanes = 64
    dilation = 1
    base_width = width_per_group
    conv1 = Conv2d(3, inplanes, 7; stride=2, padding=3, bias=false)
    bn1 = norm_layer(inplanes)
    layer1, inplanes = make_layer(block, inplanes, 64, block_nums[1], base_width)
    layer2, inplanes = make_layer(block, inplanes, 128, block_nums[2], base_width; stride=2)
    layer3, inplanes = make_layer(block, inplanes, 256, block_nums[3], base_width; stride=2)
    layer4, inplanes = make_layer(block, inplanes, 512, block_nums[3], base_width; stride=2)
    fc = Linear(512 * expansion(block), num_classes)
    resnet = ResNet(conv1, bn1, layer1, layer2, layer3, layer4, fc)
    walkstruct(resnet) do s
        if s isa Conv2d
            init_kaiming_normal!(s.W; mode=:fan_out, nonlinearity=:relu)
        elseif s isa BatchNorm2d
            init_constant!(s.gamma, 1.0)
            init_constant!(s.beta, 0.0)
        end
    end
    if zero_init_residual
        walkstruct(resnet) do s
            if s isa Bottleneck
                init_constant!(s.bn3.gamma, 0.0)
            elseif s isa BasicBlock
                init_constant!(s.bn2.gamma, 0.0)
            end
        end
    end
    return resnet
end


function (m::ResNet)(x::AbstractArray{T, 4}) where T
    x = m.conv1(x)
    x = m.bn1(x)
    x = relu.(x)
    x = maxpool2d(x, 3, stride=2, padding=1)

    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)

    # using maxpool2d instead of AdaptiveAvgPool since later isn't implemented yet
    # AdaptiveXXXPool computes kernel size automatically based on output size
    # (which is (1,1) in PyTorch implementation)
    # instead we compute kernel size ourselves as size(x)[1:2] .+ (1, 1) .- (1, 1) ./ (1, 1)
    # which is just (4, 4)
    x = maxpool2d(x, (4, 4))
    x = dropdims(x, dims=(1,2))
    x = m.fc(x)

    return x
end
