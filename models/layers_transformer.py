import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple

class PositionalEncoding(nn.Module):
    """Fixed sinusoid positional encoding to provide sequence order information."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class GaussianLayer(nn.Module):
    """Gaussian layer to map hidden representations to latent distributions."""
    def __init__(self, input_size: int, dim: int):
        super().__init__()
        self.mulayer = nn.Linear(input_size, dim)
        self.logvar_layer = nn.Linear(input_size, dim)

    def forward(self, input_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mulayer(input_layer)
        logvar = self.logvar_layer(input_layer)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return mu, logvar, sample

class TransformerSeqEncoder(nn.Module):
    """Transformer-based encoder for z2 (latent sequence variable)."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 8):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        self.pos_encoder = PositionalEncoding(hus)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z2_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer(x)
        # Sequence summary via mean pooling
        seq_summary = output.mean(dim=1)
        return self.z2_gauss_layer(seq_summary)

class TransformerSegEncoder(nn.Module):
    """Transformer-based encoder for z1 (latent segment variable) conditioned on z2."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 8):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.embedding = nn.Linear(input_size, hus)
        self.pos_encoder = PositionalEncoding(hus)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hus, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.z1_gauss_layer = GaussianLayer(hus, output_size)
        # self.attn_pool = nn.Linear(hus, 1) # to learn attention weights for pooling 

    def forward(self, x: torch.Tensor, lat_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, T, _ = x.shape
        # Condition z1 on z2 by concatenating z2 to every frame of x
        lat_seq = lat_seq.unsqueeze(1).repeat(1, T, 1)
        x_combined = torch.cat([x, lat_seq], dim=-1)
        
        x_emb = self.embedding(x_combined)
        x_emb = self.pos_encoder(x_emb)
        output = self.transformer(x_emb)
        # Segment summary via mean pooling
        seg_summary = output.mean(dim=1)
        # attn_logits = self.attn_pool(output) # [batch, T, 1]
        # attn_weights = torch.softmax(attn_logits, dim=1) # Normalize weights to sum to 1
        # seg_summary = torch.sum(output * attn_weights, dim=1) # Weighted average
        return self.z1_gauss_layer(seg_summary)

class TransformerDecoder(nn.Module):
    """Transformer-based decoder using z1 and z2 as memory for reconstruction."""
    def __init__(self, input_size: int, output_size: int, hus: int = 256, n_layers: int = 2, nhead: int = 8):
        super().__init__()
        assert hus % nhead == 0, f"d_model (hus={hus}) must be divisible by nhead ({nhead})"
        
        self.d_model = hus
        self.embedding = nn.Linear(input_size, hus)
        self.pos_encoder = PositionalEncoding(hus)
        
        # We use a TransformerDecoder to attend to the latent 'memory'
        decoder_layer = nn.TransformerDecoderLayer(d_model=hus, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.dec_gauss_layer = GaussianLayer(hus, output_size)

    def forward(self, lat_seg: torch.Tensor, lat_seq: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = lat_seg.shape[0]
        
        # Combine z1 and z2 into the memory context
        z1_z2 = torch.cat([lat_seg, lat_seq], dim=-1).unsqueeze(1) # (BS, 1, input_size)
        memory = self.embedding(z1_z2) # (BS, 1, d_model)
        
        # Create queries for the desired output sequence length
        # Using a zero-initialized query sequence + Positional Encoding
        queries = torch.zeros(batch_size, seq_len, self.d_model).to(lat_seg.device)
        queries = self.pos_encoder(queries)
        
        # The decoder attends to the 'memory' (the latents) at every step
        output = self.transformer_decoder(queries, memory)
        
        # Map back to the original feature space (x_mu, x_logvar, x_sample)
        x_mu, x_logvar, x_rec = self.dec_gauss_layer(output)
        return x_mu, x_logvar, x_rec