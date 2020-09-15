mutable struct MyModel
    linear::Linear
end

(m::MyModel)(x::AbstractArray) = m.linear(x)

my_model_loss(m::MyModel, x::AbstractArray) = sum(m(x))

@testset "optim" begin
    # not every parameter update will lead to descreased loss
    # but at least we can check that parameters are actually changed
    m = MyModel(Linear(5, 4)); x = rand(5, 10);
    old_m = deepcopy(m); old_x = copy(x)
    _, g = grad(my_model_loss, m, x)
       
    # SGD
    update!(SGD(0.1; momentum=0.5), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(SGD(0.1; momentum=0.5), x, g[2])
    @test old_x != x

    # RMSprop
    update!(RMSprop(), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(RMSprop(), x, g[2])
    @test old_x != x

    # Adam
    update!(Adam(), m, g[1])
    @test old_m.linear.W != m.linear.W
    update!(Adam(), x, g[2])
    @test old_x != x
end
