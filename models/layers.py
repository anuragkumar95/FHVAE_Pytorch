import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from typing import List, Tuple
import numpy as np

class PositionalEncoding(nn.Module):
    """Fixed sinusoid positional encoding to provide sequence order information."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VariableLSTMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(in_dim, out_dim, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.num_layers=num_layers

    def forward(self, x, ):
        output, (h, _) = self.lstm(x)
        return output, h


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
        self.lstm = VariableLSTMLayer(input_size, self.hus, num_layers=n_LSTM_layers)
        self.z1_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor):
        _, T, _ = x.shape
        lat_seq = lat_seq.unsqueeze(1).repeat(1, T, 1)
        x_z2 = torch.cat([x, lat_seq], dim=-1)
        _, hid = self.lstm(x_z2)
        z1_mu, z1_logvar, z1_sample = self.z1_gauss_layer(hid[-1, :, :])
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
        self.lstm = VariableLSTMLayer(input_size, hus, num_layers=n_LSTM_layers)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x):
        _, hid = self.lstm(x)
        z2_mu, z2_logvar, z2_sample = self.z2_gauss_layer(hid[-1, :, :])
        return z2_mu, z2_logvar, z2_sample

class JointLatentSeqEncoder(nn.Module):
    """Pre-stochastic layer encoder for z2 (latent sequence variable)

    Args:
        input_size: Size of first layer input
        hus:        List of numbers of layer hidden units

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, aud_in_ch: int, eeg_in_ch: int, output_size: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            hus = 1024
        self.lstm_aud = VariableLSTMLayer(aud_in_ch, hus, num_layers=n_LSTM_layers)
        self.lstm_eeg = VariableLSTMLayer(eeg_in_ch, hus, num_layers=n_LSTM_layers)
        self.z2_gauss_layer = GaussianLayer(hus * 2, output_size)

    def forward(self, x_aud, x_eeg):
        """
        Potentially audio is more richer in information and thus eeg might not inform the joint distribution. 
        Apply random masking where one modality is masked (zeroed_out) to make sure that eeg does equal amount of 
        work. 
        """
        _, hid_aud = self.lstm_aud(x_aud)
        _, hid_eeg = self.lstm_eeg(x_eeg)
        hid = torch.cat([hid_aud[-1, :, :], hid_eeg[-1, :, :]], dim=-1)
        z2_mu, z2_logvar, z2_sample = self.z2_gauss_layer(hid)
        return z2_mu, z2_logvar, z2_sample
    
class JointLatentSeqEncoder2(nn.Module):
    """Pre-stochastic layer encoder for z2 (latent sequence variable)

    Args:
        input_size: Size of first layer input
        hus:        List of numbers of layer hidden units

    Returns:
        out: Concatenation of hidden states of all layers

    """

    def __init__(self, aud_in_ch: int, eeg_in_ch: int, output_size: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            hus = 1024
        self.lstm_aud = VariableLSTMLayer(aud_in_ch, hus, num_layers=n_LSTM_layers)
        self.lstm_eeg = VariableLSTMLayer(eeg_in_ch, hus, num_layers=n_LSTM_layers)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x_aud, x_eeg):
        """
        Potentially audio is more richer in information and thus eeg might not inform the joint distribution. 
        Apply random masking where one modality is masked (zeroed_out) to make sure that eeg does equal amount of 
        work. 
        """
        _, hid_aud = self.lstm_aud(x_aud)
        _, hid_eeg = self.lstm_eeg(x_eeg)
        #hid = torch.cat([hid_aud[-1, :, :], hid_eeg[-1, :, :]], dim=-1)
        z2_mu_aud, z2_logvar_aud, z2_sample_aud = self.z2_gauss_layer(hid_aud[-1, :, :])
        z2_mu_eeg, z2_logvar_eeg, z2_sample_eeg = self.z2_gauss_layer(hid_eeg[-1, :, :])
        z2_mu = (z2_mu_aud, z2_mu_eeg)
        z2_logvar = (z2_logvar_aud, z2_logvar_eeg)
        z2_sample = (z2_sample_aud, z2_sample_eeg)
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
    
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # [Batch, Seq_len, 1]
        scores = self.attention_weights(x)
        weights = F.softmax(scores, dim=1)
        
        # [Batch, 1, Seq_len] @ [Batch, Seq_len, d_model] -> [Batch, 1, d_model]
        pooled_output = torch.bmm(weights.transpose(1, 2), x)
        return pooled_output.squeeze(1), weights

class TransformerSeqEncoder(nn.Module):
    """Transformer-based encoder for z2 (latent sequence variable)."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_in = self.embedding(x)
        out = self.transformer(x_in).mean(1)
        return self.z2_gauss_layer(out)
    
class TransformerSegEncoder(nn.Module):
    """Transformer-based encoder for z1 (latent segment variable) conditioned on z2."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        self.pos_encoder = PositionalEncoding(hus)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z1_gauss_layer = GaussianLayer(hus, output_size)
        self.attn_pool = AttentionPooling(hus)

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, T, _ = x.shape
        
        # Condition z1 on z2 by concatenating z2 to every frame of x
        lat_seq = lat_seq.unsqueeze(1).repeat(1, T, 1)
        x_z2 = torch.cat([x, lat_seq], dim=-1)
        
        x_emb = self.pos_encoder(self.embedding(x_z2))
        out = self.transformer(x_emb)
        out, _ = self.attn_pool(out)

        return self.z1_gauss_layer(out)


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
            h, _ = h_state
            x_t_mu, x_t_logvar, x_t_sample = self.dec_gauss_layer(h[-1, :, :])
            x_mu.append(x_t_mu.unsqueeze(1))
            x_logvar.append(x_t_logvar.unsqueeze(1))
            x_rec.append(x_t_sample.unsqueeze(1))
        x_mu = torch.cat(x_mu, dim=1)
        x_logvar = torch.cat(x_logvar, dim=1)
        x_rec = torch.cat(x_rec, dim=1)
        return x_mu, x_logvar, x_rec


