import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from typing import List
import numpy as np

# class VariableLSTMLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_layers=1):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.lstm = nn.LSTM(in_dim, out_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
#         self.num_layers=num_layers

#     def forward(self, x, ):
#         output, (h, _) = self.lstm(x)
#         return output, h


class LatentSegEncoder(nn.Module):
    """Pre-stochastic layer encoder for z1 (latent segment variable)

    Args:
        input_size: Size of input to first layer
        x:          Tensor of shape (bs, T, F) (z1)
        lat_seq:    Latent sequence variable (z2)
        hus:        List of numbers of FC layer hidden units

    Returns:
        out: last FC layer output

    """

    def __init__(self, input_size: int, output_size: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            self.hus = 1024
        else:
            self.hus = hus
        self.num_layers = n_LSTM_layers
        self.lstm = nn.LSTM(input_size, hus, num_layers=n_LSTM_layers, bidirectional=False, batch_first=True)
        self.z1_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor):
        _, T, _ = x.shape
        lat_seq = lat_seq.unsqueeze(1).repeat(1, T, 1)
        x_z2 = torch.cat([x, lat_seq], dim=-1)

        _, (h, _) = self.lstm(x_z2)
        z1_mu, z1_logvar, z1_sample = self.z1_gauss_layer(h[-1, :, :])
        return z1_mu, z1_logvar, z1_sample


class LatentSeqEncoder(nn.Module):
    """Pre-stochastic layer encoder for z2 (latent sequence variable)

    Args:
        input_size: Size of first layer input
        hus:        List of numbers of layer hidden units

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, input_size: int, output_size: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            hus = 1024
        self.lstm = nn.LSTM(input_size, hus, num_layers=n_LSTM_layers, bidirectional=False, batch_first=True)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        z2_mu, z2_logvar, z2_sample = self.z2_gauss_layer(h[-1, :, :])
        return z2_mu, z2_logvar, z2_sample


class GaussianLayer(nn.Module):
    """Gaussian layer

    Args:
        input_size:  Size of input to first layer
        dim:         Dimension of output latent variables
        input_layer: Input layer

    Returns:
        Average, log variance, and a sample from the gaussian

    """

    def __init__(self, input_size: int, dim: int):
        super().__init__()
        self.mulayer = nn.Linear(input_size, dim)
        self.logvar_layer = nn.Linear(input_size, dim)

    def forward(self, input_layer: torch.Tensor):
        mu = self.mulayer(input_layer)
        logvar = self.logvar_layer(input_layer)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return mu, logvar, sample


class LSTMDecoder(nn.Module):
    """Pre-stochastic layer decoder

    Args:
        input_size: Size of input data
        hus:        List of hidden units per fully-connected layer
        lat_seg:    Latent segment Tensor (z1)
        lat_seq:    Latent sequence Tensor (z2)

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, input_size: int, output_size: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            hus = 1024
        self.hid_dim = hus
        self.lstm = nn.LSTM(input_size, hus, num_layers=n_LSTM_layers, bidirectional=False, batch_first=True)
        self.dec_gauss_layer = GaussianLayer(hus, output_size)
        self.num_layers = n_LSTM_layers

    def init_hidden(self, batch_size, dev=None, dtype=None):
        h = Variable(torch.zeros(self.num_layers, batch_size, self.hid_dim))
        c = Variable(torch.zeros(self.num_layers, batch_size, self.hid_dim))
        if dev is not None:
            h = h.to(dev)
            c = c.to(dev)
        if dtype is not None:
            h = h.to(dtype)
            c = c.to(dtype)
        return (h, c)

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor, seq_len: int):
        x_rec = []
        x_mu = []
        x_logvar = []

        BS = lat_seg.shape[0]
        dev = lat_seg.device
        dtype = lat_seq.dtype

        h_state = self.init_hidden(BS, dev, dtype)
        z1_z2 = torch.cat([lat_seg, lat_seq], -1).unsqueeze(1)
        
        for _ in range(seq_len):
            _, h_state = self.lstm(z1_z2, h_state)
            x_t_mu, x_t_logvar, x_t_sample = self.dec_gauss_layer(h_state[0])
            x_mu.append(x_t_mu[-1, :, :].unsqueeze(1))
            x_logvar.append(x_t_logvar[-1, :, :].unsqueeze(1))
            x_rec.append(x_t_sample[-1, :, :].unsqueeze(1))
        x_mu = torch.cat(x_mu, dim=1)
        x_logvar = torch.cat(x_logvar, dim=1)
        x_rec = torch.cat(x_rec, dim=1)
        return x_mu, x_logvar, x_rec


   