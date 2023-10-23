import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from customrnns.rnn import LSTM


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size, num_layers, batch_first=False):
    if batch_first:
        inp = torch.randn(batch, seq_len, input_size)
    else:
        inp = torch.randn(seq_len, batch, input_size)

    custom_lstm = LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
    for layer in range(num_layers):
        custom_params = list(custom_lstm.parameters())[4 * layer: 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer],
                                            custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)

    custom_out, custom_out_state = custom_lstm(inp)
    lstm_out, lstm_out_state = lstm(inp)

    assert (custom_out - lstm_out).abs().max() < 1e-5
    assert (custom_out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_rnn_packed_sequence(input_size, hidden_size, num_layers=1, batch_first=False):
    sequences = [
        [1., 2., 3.],
        [4., 5.],
        [6., 7., 8., 9.],
    ]

    # Padding
    padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)
    padded_sequences = torch.unsqueeze(padded_sequences, -1)

    # Create PackedSequence: Tensor(9,1) [6,1,4,7,2,5,8,3,9]; Tensor(4,) [3,3,2,1]; Tensor(3,) [2,0,1]
    packed_input = pack_padded_sequence(padded_sequences, [len(seq) for seq in sequences], batch_first=True,
                                        enforce_sorted=False)

    custom_lstm = LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
    for layer in range(num_layers):
        custom_params = list(custom_lstm.parameters())[4 * layer: 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer],
                                            custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)

    # lstm_out: Tensor(9,hidden_size); Tensor(4,); Tensor(3,) [2,0,1]
    # lstm_out_state: (Tensor(1,3,hidden_size), Tensor(1,3,hidden_size))
    lstm_out, lstm_out_state = lstm(packed_input)
    custom_out, custom_out_state = custom_lstm(packed_input)

    # print(f'custom_out: {custom_out.data.shape}')
    # print(custom_out.data)
    # print(f'lstm_out: {lstm_out.data.shape}')
    # print(lstm_out.data)

    assert (custom_out.data - lstm_out.data).abs().max() < 1e-5
    assert (custom_out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


test_script_stacked_rnn(5, 2, 3, 7, 4)
test_script_stacked_rnn(5, 2, 3, 7, 4, True)
test_script_stacked_rnn_packed_sequence(1, 3, 2)
