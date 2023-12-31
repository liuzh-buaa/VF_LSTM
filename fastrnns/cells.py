"""
    https://github.com/pytorch/pytorch/blob/e9ef087d2d12051341db485c8ac64ea64649823d/benchmarks/fastrnns/cells.py
"""
import torch


def milstm_cell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = (alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias)

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def flat_lstm_cell(input, hx, cx, w_ih, w_hh, b_ih, b_hh):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell(igates, hidden, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def premul_lstm_cell_no_bias(igates, hidden, w_hh, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden
    gates = igates + torch.mm(hx, w_hh.t()) + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = torch.mm(input, w_ih.t()) + b_ih
    gh = torch.mm(hidden, w_hh.t()) + b_hh
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def rnn_relu_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.relu(igates + hgates)


def rnn_tanh_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    igates = torch.mm(input, w_ih.t()) + b_ih
    hgates = torch.mm(hidden, w_hh.t()) + b_hh
    return torch.tanh(igates + hgates)
