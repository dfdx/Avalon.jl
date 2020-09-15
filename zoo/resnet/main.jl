using Avalon

include("model.jl")
include("imagefolder.jl")


# this should point to a real folder containing subfolders with images
# each subfolder name will be taken as class name, while contained images
# will be used for training
# images can have any size, but will be resized to (100, 100) during loading
const IMAGE_DATASET_PATH = "~/data/food-101-sample"
# number of subfolders with images
const NUM_CLASSES = 5


# TODO:
# 4. load pretrained net
# 5. make wrappers like resnet18(), etc.


################################################################################
#                         TRAINING FROM SCRATCH                                #
################################################################################


# note that this method doesn't do validation or testing, but is only useful to make sure
# value of loss function is reduced over time
function Avalon.fit!(m::ResNet, dataset::ImageFolder, loss_fn;
              n_epochs=10, batch_size=100, opt=Adam(;lr=1e-3), device=CPU(), report_every=1)
    f = (m, x, y) -> loss_fn(m(x), y)
    num_batches = length(dataset) // batch_size
    for epoch in 1:n_epochs
        epoch_loss = 0
        for (i, (x, y)) in enumerate(Avalon.batchiter(dataset, sz=batch_size))
            x = to_device(device, copy(x))
            y = to_device(device, copy(y))
            loss, g = grad(f, m, x, y)
            update!(opt, m, g[1])
            epoch_loss += loss
            println("  batch loss = $loss")
        end
        if epoch % report_every == 0
            println("Epoch $epoch: avg_loss=$(epoch_loss / num_batches)")
        end
    end
    return m
end


function train_from_scretch()
    device = best_available_device()
    dataset = ImageFolder(expanduser(IMAGE_DATASET_PATH), transform=img -> imresize(img, (100, 100)))
    block = Bottleneck
    block_nums = [2, 2, 2, 2]
    stride = 1
    m = ResNet(block, block_nums, zero_init_residual=true, num_classes=NUM_CLASSES) |> device
    loss_fn = CrossEntropyLoss()
    fit!(m, dataset, loss_fn; device=device, batch_size=5)
end


################################################################################
#                         USING PRETRAINED MODEL                               #
################################################################################

# Will be added once ONNX import/export is implemented
