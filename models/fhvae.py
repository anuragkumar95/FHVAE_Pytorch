import torch
import torch.nn as nn
import numpy as np
from typing import List
import math
import torch.nn.functional as F

from models.layers import (
    LatentSegEncoder,
    LatentSeqEncoder,
    LSTMDecoder,
)

from models.layers_transformer import TransformerSeqEncoder, TransformerSegEncoder, TransformerDecoder

class FHVAE(nn.Module):
    def __init__(
        self,
        input_size,
        n_seqs, 
        z1_hus=128,
        z2_hus=128,
        z1_dim=16,
        z2_dim=16,
        x_hus=128,
        n_LSTM_layers=1,
        n_layers=2,
        nhead=8,
    ):
        super().__init__()
        self.model = "fhvae"

        # priors
        self.pz1 = [0.0, np.log(1.0 ** 2).astype(np.float32)]
        self.pmu2 = [0.0, np.log(1.0 ** 2).astype(np.float32)]

        # encoder/decoder arch
        self.z1_hus = z1_hus
        self.z2_hus = z2_hus
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.x_hus = x_hus

        #Lookup table
        self.mu2_table = nn.Embedding(num_embeddings=n_seqs, embedding_dim=z2_dim)       

        # using LSTM for both encoders and decoder as per original FHVAE paper
        
        # self.z2_encoder = LatentSeqEncoder(
        #     input_size=input_size, 
        #     output_size=self.z2_dim,
        #     hus=self.z2_hus, 
        #     n_LSTM_layers=n_LSTM_layers
        # )
        
        # self.z1_encoder = LatentSegEncoder(
        #     input_size=input_size+self.z2_dim, 
        #     output_size=self.z1_dim,
        #     hus=self.z1_hus, 
        #     n_LSTM_layers=n_LSTM_layers
        # )
       
        self.decoder = LSTMDecoder(
            input_size=self.z1_dim+self.z2_dim, 
            output_size=input_size,
            hus=self.x_hus, 
            n_LSTM_layers=n_LSTM_layers
        )


        # replacing the LSTM encoder with transformer-based ones
        self.z2_encoder = TransformerSeqEncoder(
            input_size=input_size, 
            output_size=z2_dim,
            hus=z2_hus, 
            n_layers=n_layers,
            nhead=nhead
        )

        self.z1_encoder = TransformerSegEncoder(
            input_size=input_size + z2_dim,
            output_size=z1_dim,
            hus=z1_hus, 
            n_layers=n_layers,
            nhead=nhead
        )
        
        # self.decoder = TransformerDecoder(
        #     input_size=z1_dim + z2_dim, 
        #     output_size=input_size,
        #     hus=x_hus, 
        #     n_layers=n_layers,
        #     nhead=nhead
        # )

        self.loss = nn.CrossEntropyLoss()

    def log_gauss(self, x, mu=0.0, logvar=0.0):
        """Compute log N(x; mu, exp(logvar))"""
        if isinstance(logvar, np.float32) or isinstance(logvar, float):
            logvar = torch.tensor(logvar).to(x.device)
    
        return -0.5 * (
            np.log(2 * np.pi) + logvar + (torch.pow(x - mu, 2) / torch.exp(logvar))
        )

    def kld(self, p_mu, p_logvar, q_mu, q_logvar):
        """Compute D_KL(p || q) of two Gaussians"""
        return -0.5 * (
            1
            + p_logvar
            - q_logvar
            - (torch.pow(p_mu - q_mu, 2) + torch.exp(p_logvar)) / np.exp(q_logvar)
        )
    
    def extract_z2(self, x):
        """Extract z2 latent features"""
        return self.z2_encoder(x)
    
    def extract_z1(self, x):
        """Extract z1 latent features"""
        _, _, z2_sample = self.extract_z2(x)
        return self.z1_encoder(x, z2_sample)
    
    def extract_latents(self, x):
        """Extract z1, z2 latent features"""
        z2_mu, z2_logvar, z2_sample = self.extract_z2(x)
        z1_mu, z1_logvar, z1_sample = self.extract_z1(x)
        return {
            'z2': {'mu':z2_mu, 'logvar':z2_logvar, 'sample':z2_sample},
            'z1': {'mu':z1_mu, 'logvar':z1_logvar, 'sample':z1_sample},
        }

    def reconstruct_latents(self, z1_sample, z2_sample, seq_len):
        x_mu, x_logvar, x_sample = self.decoder(z1_sample, z2_sample, seq_len)
        return {
            'mu': x_mu, 
            'logvar': x_logvar,
            'sample': x_sample
        } 
    
    def est_mu2_lookup(self, z2, N, p_var_z2, p_var_mu2):
        mu2 = z2.sum(0) / (N + (p_var_z2 / p_var_mu2))
        mu2 = mu2.repeat(N, 1)
        return mu2

    def forward(
        self, x: torch.Tensor, mu_idx: torch.Tensor, num_seqs: int, num_segs: int, mode: str = 'train'
    ):
        """Forward pass through the network

        Args:
            x:        Input data
            mu_idx:   Int tensor of shape (bs,). Index for mu2_table
            num_seqs: Size of mu2 lookup table
            num_segs: Number of audio segments

        Returns:
            Variational lower bound and discriminative loss

        """
        # Get z2 latents
        z2_mu, z2_logvar, z2_sample = self.z2_encoder(x)
        qz2_x = [z2_mu, z2_logvar]

        # define constant prior variance for z2
        pz2_logvar = np.log(0.5 ** 2).astype(np.float32)

        if mode == 'train':
            #mu2_table, mu2 = self.mu2_lookup(mu_idx, self.z2_dim, num_seqs, device=x.device)
            mu2 = self.mu2_table(mu_idx)
        else:
            # mu2 lookup for unseen seq
            # NOTE: during val/test, we send one whole sequence divided into its segments as one batch
            # thus, z2_mu should be summed along the first dim 0.

            # mu2 = z2_mu.sum(0) / (x.shape[0] + (np.exp(pz2[1]) / np.exp(self.pmu2[1])))
            # mu2 = mu2.repeat(x.shape[0], 1)
            # pz2[0] = mu2

            # updated version:
            mu2 = z2_mu.sum(0) / (x.shape[0] + (np.exp(pz2_logvar) / np.exp(self.pmu2[1])))
            mu2 = mu2.repeat(x.shape[0], 1)
            
        # z2 prior
        # pz2 = [mu2, np.log(0.5 ** 2).astype(np.float32)]

        pz2 = [mu2, pz2_logvar]

        # Get z1 latents
        if mode!= 'train':
            z2_sample = z2_mu
        z1_mu, z1_logvar, z1_sample = self.z1_encoder(x, z2_sample)
        qz1_x = [z1_mu, z1_logvar]

        # Decode for reconstruction
        if mode != 'train':
            z1_sample = z1_mu
        x_mu, x_logvar, x_sample = self.decoder(z1_sample, z2_sample, seq_len=x.shape[1])
        px_z = [x_mu, x_logvar]

        # variational lower bound
        log_pmu2 = torch.sum(
            self.log_gauss(mu2, self.pmu2[0], self.pmu2[1]), dim=1
        )

        neg_kld_z2 = -1 * torch.sum(
            self.kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), dim=1
        )
        neg_kld_z1 = -1 * torch.sum(
            self.kld(qz1_x[0], qz1_x[1], self.pz1[0], self.pz1[1]), dim=1
        )
        log_px_z = torch.sum(
            self.log_gauss(x, px_z[0], px_z[1]), dim=(1,2)
        )
        
        num_segs = num_segs.to(x.device)
        lower_bound = log_px_z + neg_kld_z1 + neg_kld_z2 + (log_pmu2 / num_segs)
        
        log_qy = -1
        if mode == 'train':
            # discriminative loss only during training
            logits = torch.unsqueeze(qz2_x[0], 1) - torch.unsqueeze(self.mu2_table.weight, 0)
            logits = -1 * torch.pow(logits, 2) / (2 * np.exp(pz2[1]))
            logits = torch.sum(logits, dim=-1)
            log_qy = -self.loss(input=logits, target=mu_idx)
        return lower_bound, log_qy, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_sample
    
    def generate(self, x):
        """
        Function that reconstructs the input. Do not use this for training.
        ARGS:
            x: torch.Tensor of shape (B, T, C)
        """
        assert len(x.shape) == 3, f"X shape should be (B, T, C)"
        latents = self.extract_latents(x)
        z1 = latents['z1']['mu']
        z2 = latents['z2']['mu']
        return self.reconstruct_latents(z1, z2, seq_len=x.shape[1])