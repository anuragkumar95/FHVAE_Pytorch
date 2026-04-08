import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from typing import List, Tuple


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

    def forward(self, x: torch.Tensor, weight: torch.Tensor = 1) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + weight * self.pe[:, :x.size(1), :]
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

    def forward(self, x: torch.Tensor, lat_seqs: List[torch.Tensor]):
        _, T, _ = x.shape
        lat_seq = [lat_seq.unsqueeze(1).repeat(1, T, 1) for lat_seq in lat_seqs]
        x_z2 = torch.cat([x] + lat_seq, dim=-1)
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
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 4, norm_first: bool = False, pool: bool = True):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

        if pool:
            self.attn_pool = AttentionPooling(hus)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_in = self.embedding(x)
        
        out = self.transformer(x_in)
        if self.pool:
            out, _ = self.attn_pool(out)
        
        return self.z2_gauss_layer(out)

    
class TransformerSegEncoder(nn.Module):
    """Transformer-based encoder for z1 (latent segment variable) conditioned on z2."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 4, norm_first: bool = False, pool: bool = True, pos_enc: bool =True):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        if pos_enc:
            self.pos_encoder = PositionalEncoding(hus)
        else:
            self.pos_encoder = None
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True, norm_first=norm_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z1_gauss_layer = GaussianLayer(hus, output_size)
        if pool:
            self.attn_pool = AttentionPooling(hus)
        self.pool = pool

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:       Input signal (B, D)
        lat_seq: List of global seq latents
        """
        _, T, _ = x.shape
        
        # Condition z1 on z2 by concatenating z2 to every frame of x
        lat_seq = lat_seq.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([x, lat_seq], dim=-1)
        
        x_emb = self.embedding(x)
        if self.pos_encoder:
            x_emb = self.pos_encoder(x_emb)

        out = self.transformer(x_emb)
        if self.pool:
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
        dtype = lat_seg.dtype

        h_state = self.init_hidden(BS, dev, dtype)
        z1_z2 = torch.cat([lat_seq, lat_seg], -1).unsqueeze(1)
        
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
    
class SharedLSTMDecoder(nn.Module):
    """Pre-stochastic layer decoder

    Args:
        input_size: Size of input data
        hus:        List of hidden units per fully-connected layer
        lat_seg:    Latent segment Tensor (z1)
        lat_seq:    Latent sequence Tensor (z2)

    Returns:
        out: Concatenation of hidden states of all layers

    """
    def __init__(self, input_size: int, out_eeg_ch: int, out_aud_ch: int, hus: int = None, n_LSTM_layers=1):
        super().__init__()
        if hus is None:
            hus = 1024
        self.hid_dim = hus
        self.lstm = nn.LSTM(input_size, hus, num_layers=n_LSTM_layers, bidirectional=False, batch_first=True)
        self.dec_eeg_gauss_layer = GaussianLayer(hus, out_eeg_ch)
        self.dec_aud_gauss_layer = GaussianLayer(hus, out_aud_ch)
        self.modality_embedding = nn.Embedding(2, input_size)
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

    def forward(self, mode: str, lat_seg: torch.Tensor, lat_seq: torch.Tensor, seq_len: int):
        x_rec = []
        x_mu = []
        x_logvar = []

        BS = lat_seg.shape[0]
        dev = lat_seg.device
        dtype = lat_seq.dtype

        h_state = self.init_hidden(BS, dev, dtype)
        z1_z2 = torch.cat([lat_seg, lat_seq], -1)

        mode_idx = torch.tensor(0 if mode == 'eeg' else 1, device=z1_z2.device)
        mode_emb = self.modality_embedding(mode_idx)
        z1_z2 = (z1_z2 + mode_emb).unsqueeze(1)
        
        for _ in range(seq_len):
            _, h_state = self.lstm(z1_z2, h_state)
            h, _ = h_state
            if mode == 'eeg':
                x_t_mu, x_t_logvar, x_t_sample = self.dec_eeg_gauss_layer(h[-1, :, :])
            if mode == 'aud':
                x_t_mu, x_t_logvar, x_t_sample = self.dec_aud_gauss_layer(h[-1, :, :])
            x_mu.append(x_t_mu.unsqueeze(1))
            x_logvar.append(x_t_logvar.unsqueeze(1))
            x_rec.append(x_t_sample.unsqueeze(1))

        x_mu = torch.cat(x_mu, dim=1)
        x_logvar = torch.cat(x_logvar, dim=1)
        x_rec = torch.cat(x_rec, dim=1)
        return x_mu, x_logvar, x_rec
        
    
class TransformerDecoder(nn.Module):
    """Pre-stochastic transformer decoder

    Args:
        latent_dim:       z1 or z2 dimension
        output_channels:  output dimension
        seq_len:          length of seq to reconstruct
        d_model:          hidden dim for attention

    Returns:
        out: reconstructed signal distribution. (mu, logvar, sample)

    """
    def __init__(self, latent_dim, output_channels, seq_len, d_model=256, nhead=8, n_layers=3):
        super().__init__()
        self.seq_len = seq_len
        
        self.input_proj_z2 = nn.Linear(latent_dim, d_model)
        self.input_proj_z1 = nn.Linear(latent_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 512)
    
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.output_layer = GaussianLayer(d_model, output_channels)

    def forward(self, z1, z2, seq_len=None):
        """
        z1: [B, seq_len, z1_dim]
        z2: [B, z2_dim]
        """
        if seq_len is None:
            seq_len = z1.shape[1]
      
        z1 = self.input_proj_z1(z1)
        z2 = self.input_proj_z2(z2).unsqueeze(1) # [B, seq_len, d_model]
        z_inp = torch.cat([z2, z1], dim=1) # [B, 1 + seq_len, d_model]
        z_inp = self.pos_encoder(z_inp)
        
        output = self.transformer_decoder(z_inp) # [B, 1 + seq_len, d_model]
        return self.output_layer(output[:, 1:, :]) # [B, seq_len, output_channels]
        

class FiLMBlock(nn.Module):
    """
    A Residual MLP block that uses a subject embedding to 
    scale (gamma) and shift (beta) the hidden activations.
    """
    def __init__(self, hidden_dim, subject_emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.film_gen = nn.Linear(subject_emb_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x, sub_emb):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        
        film_params = self.film_gen(sub_emb)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        out = gamma * out + beta
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        
        return self.activation(out + residual)

class ResNetSpeakerClassifier(nn.Module):
    def __init__(self, z3_dim, num_speakers, hidden_dim=256, sub_emb_dim=64, num_blocks=3):
        super().__init__()
      
        self.input_proj = nn.Linear(z3_dim, hidden_dim)
        
        # Stack of Residual FiLM blocks
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, sub_emb_dim) for _ in range(num_blocks)
        ])
        
        # Speaker Classification Head
        self.classifier_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drop = nn.Dropout(0.1)
        self.classifier_out = nn.Linear(hidden_dim // 2, num_speakers)


    def forward(self, z3, sub_emb):
        """
        Args:
            z3: [batch, z3_dim] latents from FHVAE
            sub_idx: [batch] long tensor of subject indices
        """
        # Project FHVAE latents into the classifier's hidden space
        x = self.input_proj(z3)
        
        # Pass through ResNet blocks with Subject-Conditioning and classify
        for block in self.blocks:
            x = block(x, sub_emb)

        # Classification
        x = self.classifier_proj(x)
        x = self.drop(F.gelu(x))
        logits = self.classifier_out(x)
        
        return logits

