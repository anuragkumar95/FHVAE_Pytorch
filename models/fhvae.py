import torch
import torch.nn as nn
import numpy as np
from typing import List
import math
import torch.nn.functional as F
from torch.autograd import Variable
from models.layers import (
    LatentSegEncoder,
    LatentSeqEncoder,
    LSTMDecoder,
)

class FHVAE(nn.Module):
    def __init__(
        self,
        input_size,
        z1_hus=128,
        z2_hus=128,
        z1_dim=16,
        z2_dim=16,
        x_hus=128,
        n_LSTM_layers=1,
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

        self.z1_encoder = LatentSegEncoder(
            input_size=input_size+self.z1_dim, 
            output_size=self.z1_dim,
            hus=self.z1_hus, 
            n_LSTM_layers=n_LSTM_layers
        )
        self.z2_encoder = LatentSeqEncoder(
            input_size=input_size, 
            output_size=self.z2_dim,
            hus=self.z2_hus, 
            n_LSTM_layers=n_LSTM_layers
        )
        self.decoder = LSTMDecoder(
            input_size=self.z1_dim+self.z2_dim, 
            output_size=input_size,
            hus=self.x_hus, 
            n_LSTM_layers=n_LSTM_layers
        )
        self.loss = nn.CrossEntropyLoss()

    def mu2_lookup(
        self, mu_idx: torch.Tensor, z2_dim: int, num_seqs: int, init_std: float = 1.0, device=None
    ):
        """Mu2 posterior mean lookup table

        Args:
            mu_idx:   Int tensor of shape (bs,). Index for mu2_table
            z2_dim:   Z2 dimension
            num_seqs: Lookup table size
            init_std: Standard deviation for lookup table initialization

        """
        mu2_table = torch.empty([num_seqs, z2_dim]).normal_(mean=0, std=init_std)
        if device is not None:
            mu2_table = mu2_table.to(device)
        mu2_table.requires_grad = True
        mu2 = torch.gather(mu2_table, 0, torch.stack([mu_idx] * z2_dim, 1))
        return mu2_table, mu2

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
        mu2_table, mu2 = None, None
        if mode == 'train':
            mu2_table, mu2 = self.mu2_lookup(mu_idx, self.z2_dim, num_seqs, device=x.device)

        # z2 prior
        pz2 = [mu2, np.log(0.5 ** 2).astype(np.float32)]

        # Get z2 latents
        z2_mu, z2_logvar, z2_sample = self.z2_encoder(x)
        qz2_x = [z2_mu, z2_logvar]

        if mode != 'train':
            # mu2 lookup for unseen seq
            # NOTE: during val/test, we send one whole sequence divided into its segments as one batch
            # thus, z2_mu should be summed along the first dim 0.
            mu2 = z2_mu.sum(0) / (x.shape[0] + (np.exp(pz2[1]) / np.exp(self.pmu2[1])))
            mu2 = mu2.repeat(x.shape[0], 1)
            pz2[0] = mu2

        # Get z1 latents
        if mode != 'train':
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
            logits = torch.unsqueeze(qz2_x[0], 1) - torch.unsqueeze(mu2_table, 0)
            logits = -1 * torch.pow(logits, 2) / (2 * np.exp(pz2[1]))
            logits = torch.sum(logits, dim=-1)
            log_qy = self.loss(input=logits, target=mu_idx)

        return lower_bound, log_qy, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_sample
    
    def generate(self, x):
        """Function that reconstructs the input. Do not use this for training."""
        latents = self.extract_latents(x)
        z1 = latents['z1']['mean']
        z2 = latents['z2']['mean']
        return self.reconstruct_latents(z1, z2, x.shape[1])


class ExtendedFHVAE(FHVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
        mu2_table, mu2 = None, None
        if mode == 'train':
            mu2_table, mu2 = self.mu2_lookup(mu_idx, self.z2_dim, num_seqs, device=x.device)
            mu2 = mu2.unsqueeze(1)

        if self.feature_encoder is not None:
            x = self.feature_encoder(x)
            x = x.permute(0, 2, 1)  # (bs, T, F)

        # z2 prior
        pz2 = [mu2, np.log(0.5 ** 2).astype(np.float32)]

        z2_pre_out = self.z2_pre_encoder(x)
        z2_mu, z2_logvar, z2_sample = self.z2_gauss_layer(z2_pre_out)
        qz2_x = [z2_mu, z2_logvar]

        if mode != 'train' or mu2 is None:
            # mu2 lookup for unseen seq
            mu2 = z2_mu.sum(0) / (x.shape[0] + np.exp(pz2[1]))
            mu2 = mu2.repeat(x.shape[0], 1, 1)
            pz2[0] = mu2

        z1_pre_out = self.z1_pre_encoder(x, z2_sample)
        z1_mu, z1_logvar, z1_sample = self.z1_gauss_layer(z1_pre_out)
        qz1_x = [z1_mu, z1_logvar]

        x_pre_out = self.pre_decoder(z1_sample, z2_sample)
        x_mu, x_logvar, x_sample = self.dec_gauss_layer(x_pre_out)
        px_z = [x_mu, x_logvar]

        if self.feature_decoder is not None:
            x_sample = self.feature_decoder(x_sample)

        # variational lower bound
        log_pmu2 = torch.sum(
            self.log_gauss(mu2, self.pmu2[0], self.pmu2[1]), dim=(1,2)
        )
        neg_kld_z2 = -1 * torch.sum(
            self.kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), dim=(1,2)
        )
        neg_kld_z1 = -1 * torch.sum(
            self.kld(qz1_x[0], qz1_x[1], self.pz1[0], self.pz1[1]), dim=(1,2)
        )
        log_px_z = torch.sum(
            self.log_gauss(x, px_z[0], px_z[1]), dim=(1,2)
        )
        
        lower_bound = log_px_z + neg_kld_z1 + neg_kld_z2 + (log_pmu2 / num_segs)
        
        log_qy = -1
        if mode == 'train':
            # discriminative loss only during training
            qz2_x = [qz2_x[0].mean(1), qz2_x[1].mean(1)]
            logits = torch.unsqueeze(qz2_x[0], 1) - torch.unsqueeze(mu2_table, 0)
            logits = -1 * torch.pow(logits, 2) / (2 * np.exp(pz2[1]))
            logits = torch.sum(logits, dim=-1)
            log_qy = self.loss(input=logits, target=mu_idx)

        return lower_bound, log_qy, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_sample
