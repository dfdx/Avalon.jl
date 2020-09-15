@testset "RNN: vanilla" begin
    # gradient test for RNNCell
    m = RNNCell(10, 5); x = rand(10, 4); h = rand(5, 4)
    @test gradcheck((W_ih, W_hh, b_ih, b_hh, x, h) -> sum(rnn_forward(W_ih, W_hh, b_ih, b_hh, x, h)),
                    m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h)
    # smoke tests for RNN
    m = RNN(10 => 5); x_seq = rand(10, 4, 2); h = init_hidden(m, 4)
    grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h) end, m, x_seq, h)    
    grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, m, x_seq, h)
end

@testset "RNN: LSTM" begin
    # gradient test for LSTMCell
    m = LSTMCell(10, 5); x = rand(10, 4); h = rand(5, 4); c = rand(5, 4)
    @test gradcheck((W_ih, W_hh, b_ih, b_hh, x, h, c) -> begin
                    h, c = lstm_forward(W_ih, W_hh, b_ih, b_hh, x, h, c)
                    sum(h)
                    end,
                    m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h, c)    
    # smokes for LSTM
    m = LSTM(10 => 5); x_seq = rand(10, 4, 2); h, c = init_hidden(m, 4)
    grad((m, x_seq, h, c) ->
         begin h_all, h, c = m(x_seq, h, c); sum(h) end, m, x_seq, h, c)
    grad((m, x_seq, h, c) ->
         begin h_all, h, c = m(x_seq, h, c); sum(h_all) end, m, x_seq, h, c)
end

@testset "RNN: GRU" begin
    # gradient test for GRUCell
    m = GRUCell(10, 5); x = rand(10, 4); h = rand(5, 4)
    @test gradcheck((W_ih, W_hh, b_ih, b_hh, x, h) -> sum(gru_forward(W_ih, W_hh, b_ih, b_hh, x, h)),
                    m.W_ih, m.W_hh, m.b_ih, m.b_hh, x, h)
    # smoke tests for GRU
    m = GRU(10 => 5); x_seq = rand(10, 4, 2); h = init_hidden(m, 4)
    grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h) end, m, x_seq, h)    
    grad((m, x_seq, h) -> begin h_all, h = m(x_seq, h); sum(h_all) end, m, x_seq, h)
end
