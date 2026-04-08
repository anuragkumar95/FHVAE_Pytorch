import sys
import argparse
import os
import json
import random
from tqdm import tqdm
import torch
from models.fhvae import FHVAE
from torch.optim import Adam
import numpy as np
from Datasets.datasets_eeg import Joint_AUD_EEG_Dataset
from utils import (
    check_best,
)
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def set_seed(seed, rank):

    # The actual seed used by this specific process
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    
    print(f"[Rank {rank}] Seed set to {process_seed}")

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    return world_size, local_rank

# alpha/discriminative weight of 10 was found to produce best results
def loss_function(lower_bound, log_qy, alpha=10.0):
    """Discriminative segment variational lower bound

    Returns:
        Segment variational lower bound plus the (weighted) discriminative objective.

    """
    return -1 * torch.mean(lower_bound + alpha * log_qy)


def check_terminate(epoch, best_epoch, patience, epochs):
    """Checks if training should be terminated"""
    if (epoch - 1) - best_epoch > patience:
        return True
    if epoch > epochs:
        return True
    return False


class Trainer:
    def __init__(self, config, args, rank=None, world_size=None, parallel=False):

        self.config = config
        self.device = torch.device(f"cuda:{rank}")
        self.rank = rank
        self.parallel = parallel
        self.world_size = world_size
        self.start_epoch = 0
        self.epochs = config['training_args']['epochs']

        # Prepare datasets
        if config['task'] == 'eeg':
            self.train_ds = Joint_AUD_EEG_Dataset(
                **config['data_args'], split='train'
            )
            self.val_ds = Joint_AUD_EEG_Dataset(
                **config['data_args'], split='val'
            )
            self.test_ds = Joint_AUD_EEG_Dataset(
                **config['data_args'], split='test'
            )

        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        if parallel:
            self.train_sampler = DistributedSampler(self.train_ds, num_replicas=world_size, rank=rank)
            self.val_sampler = DistributedSampler(self.val_ds, num_replicas=world_size, rank=rank)
            self.test_sampler = DistributedSampler(self.test_ds, num_replicas=world_size, rank=rank)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=config['training_args']['batch_size'], 
            sampler=self.train_sampler,
            shuffle = not parallel,
            num_workers=0 if parallel else 10,
            pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=1, 
            sampler=self.val_sampler, 
            shuffle=False,
            num_workers=0
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=1, 
            sampler=self.test_sampler, 
            shuffle=False, 
            num_workers=0 
        )

        # Config model, optimizer
        self.model = FHVAE(
            **config['model_args'],
            n_seqs=self.train_ds.n_eeg_seqs
        ).to(self.device).double()

        if parallel and rank is not None:
            self.model = DDP(
                self.model, device_ids=[rank], output_device=rank
            )
        
        self.optimizer = Adam(
            self.model.parameters(), 
            **config['optimizer_args']
        )

        if args.resume_pt is not None:
            print(f"Checkpoint loading from {args.resume_pt}")
            self.load_checkpoint(args.resume_pt, map_location=self.device)

        self.accum_grad = self.config['training_args']['accum_grad']

        print(f"MODEL:\n{self.model}")
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params:,}")

        # Logging
        if self.rank == 0:
            print("Train dataloader length:", len(self.train_loader))
            print("Valid dataloader length:", len(self.val_loader))
            print("Test dataloader length:", len(self.test_loader))
            wandb.login()
            wandb.init(project='FHVAE-KUL', name=config['run_name'])
            self.exp_dir = f"./experiments/fhvae/{config['run_name']}_{args.suf}"
            os.makedirs(self.exp_dir, exist_ok=True)

    def save_checkpoint(self, **kwargs):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': kwargs.get('epoch', None),
            'val_lb': kwargs.get('val_lower_bound', None),
            'val_likelihood': kwargs.get('val_likelihood', None),
        }
        save_path = os.path.join(self.exp_dir, f"best_checkpoint_best_{kwargs.get('save_metric', 'model')}.pt")
        torch.save(save_dict, save_path)
        self.config['model_args']['n_seqs'] = self.train_ds.n_eeg_seqs
        self.config['model_args']['n_stimuli'] = self.train_ds.n_aud_seqs
        self.config['model_args']['n_speakers'] = self.train_ds.n_speaker
        self.config['model_args']['n_subjects'] = self.train_ds.n_subj
        with open(f"{self.exp_dir}/config.json", 'w') as f:
            json.dump(self.config, f)
        print(f"Checkpoint saved at {save_path}")

    def load_checkpoint(self, checkpoint_path, map_location='cpu'):
        ckpt_dir = "/".join(checkpoint_path.split('/')[:-1])
        with open(f"{ckpt_dir}/config.json", 'r') as f:
            self.config = json.load(f)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {self.start_epoch}")

    def reduce_tensor(self, tensor, world_size):
        """Sums a tensor across all GPUs and divides by world_size to get the mean."""
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
        return rt

    def train(self):

        best_val_lb = -np.inf
        best_val_likelihood = -np.inf

        self.model.double()

        for epoch in range(self.start_epoch, self.epochs):
            # training
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            train_loss = 0.0

            epoch_steps = self.config['training_args']['steps_per_epoch']
            if epoch_steps == -1:
                epoch_steps = len(self.train_loader) // self.accum_grad
            pbar = tqdm(range(min(epoch_steps, len(self.train_loader)//self.accum_grad)))  
          
            iterator = iter(self.train_loader)
            for batch_idx in pbar:
                batch_lb = 0.0
                batch_loss = 0.0
                batch_disc_loss = 0.0
                batch_log_px_z = 0.0
                batch_neg_kld_z1 = 0.0
                batch_neg_kld_z2 = 0.0
                batch_log_pmu2 = 0.0
                batch_mse = 0.0
                self.optimizer.zero_grad()
                for _ in range(self.accum_grad):
                    try:
                        (e_idx, _, efeat, _, nsegs, _, _, _, _, _) = next(iterator)
                    except StopIteration:
                        iterator = iter(self.train_loader)
                        (e_idx, _, efeat, _, nsegs, _, _, _, _, _) = next(iterator)
                    
                    features = efeat.to(self.device)
                    idxs = e_idx.to(self.device)
                    nsegs = nsegs.to(self.device)
                  
                    lower_bound, discrim_loss, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_pred = self.model(
                        x=features, mu_idx=idxs, num_segs=nsegs
                    )
                    loss = loss_function(lower_bound, discrim_loss, self.config['training_args']['alpha_dis']) / self.accum_grad
                        
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    mse = ((x_pred - features)**2).mean().item()
                    batch_lb += lower_bound.mean().item() / self.accum_grad
                    batch_log_px_z += log_px_z.mean().item() / self.accum_grad
                    batch_neg_kld_z1 += neg_kld_z1.mean().item() / self.accum_grad
                    batch_neg_kld_z2 += neg_kld_z2.mean().item() / self.accum_grad
                    batch_log_pmu2 += log_pmu2.mean().item() / self.accum_grad
                    batch_disc_loss += discrim_loss.mean().item() / self.accum_grad
                    batch_mse += mse / self.accum_grad
                    batch_loss += loss.item() / self.accum_grad

                self.optimizer.step()
                train_loss += batch_loss

                if torch.isnan(lower_bound).any():
                    print("Training diverged")
                    raise sys.exit(2)

                if self.rank == 0:
                    wandb.log({
                        "train/Loss": batch_loss / features.shape[1],
                        "train/MSE": mse,
                        "train/LowerBound": batch_lb / features.shape[1],
                        "train/Discrim_Loss": -batch_disc_loss / features.shape[1],
                        "Step": epoch * self.config['training_args']['steps_per_epoch'] + (batch_idx + 1),
                        "train/Likelihood P(x|z)": batch_log_px_z / features.shape[1],
                        "train/KL_z1": (-batch_neg_kld_z1) / features.shape[1],
                        "train/KL_z2": (-batch_neg_kld_z2) / features.shape[1],
                        "train/Log p(mu2)": batch_log_pmu2 / features.shape[1],
                    })

            train_loss /= len(self.train_loader)
            print(f"====> Train set average loss: {train_loss:.4f}")

            # eval
            self.model.eval()
            VAL_LB = 0
            VAL_MSE = 0
            VAL_PX_Z = 0
            VAL_KLD_Z1 = 0
            VAL_KLD_Z2 = 0
            VAL_KLD_Z3 = 0
            VAL_LOG_PMU2 = 0

            with torch.no_grad():
                pbar = tqdm(self.val_loader)
                batch_idx = 1
                for (e_idx, _, efeat, _, nsegs, _, _, _, _, _) in pbar:
                    
                    feature = efeat.to(self.device).squeeze(0)
                    idxs = idxs.to(self.device).squeeze(0)
                    nsegs = nsegs.to(self.device).squeeze(0)
                     
                    val_BS = 64
                    n_mini_batches = ((feature.shape[0]) // val_BS) + int((feature.shape[0]) % val_BS != 0)
                    for i in range(n_mini_batches): 
                        
                        feature_batch = feature[i*val_BS:(i+1)*val_BS]
                        idxs_batch = idxs[i*val_BS:(i+1)*val_BS]
                        nsegs_batch = nsegs[i*val_BS:(i+1)*val_BS]

                        val_lower_bound, _, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_pred = self.model(
                            feature_batch, idxs_batch, nsegs_batch, mode='val'
                        )
                    
                        VAL_LB += val_lower_bound.mean().item()/(feature_batch.shape[1] * n_mini_batches)
                        VAL_MSE += ((x_pred - feature_batch)**2).mean().item() / n_mini_batches
                        VAL_PX_Z += log_px_z.mean().item()/(feature_batch.shape[1] * n_mini_batches)
                        VAL_KLD_Z1 += (-neg_kld_z1).mean().item()/(feature_batch.shape[1] * n_mini_batches)
                        VAL_KLD_Z2 += (-neg_kld_z2).mean().item()/(feature_batch.shape[1] * n_mini_batches)
                        VAL_LOG_PMU2 += log_pmu2.mean().item()/(feature_batch.shape[1] * n_mini_batches)
                    
                    pbar.set_postfix({
                        "Epoch": epoch,
                        "Val_LB": VAL_LB / ((batch_idx+1) * feature.shape[1]) 
                    })
                    batch_idx += 1

            if self.parallel:
                self.reduce_tensor(VAL_LB, self.world_size)
                self.reduce_tensor(VAL_MSE, self.world_size)
                self.reduce_tensor(VAL_PX_Z, self.world_size)
                self.reduce_tensor(VAL_KLD_Z1, self.world_size)
                self.reduce_tensor(VAL_KLD_Z2, self.world_size)
                self.reduce_tensor(VAL_LOG_PMU2, self.world_size)

            VAL_LB /= len(self.val_loader)
            VAL_MSE /= len(self.val_loader)
            VAL_PX_Z /= len(self.val_loader)
            VAL_KLD_Z1 /= len(self.val_loader)
            VAL_KLD_Z2 /= len(self.val_loader)
            VAL_KLD_Z3 /= len(self.val_loader)
            VAL_LOG_PMU2 /= len(self.val_loader)

            if self.rank == 0:
                wandb.log({
                    "val/MSE": VAL_MSE,
                    "val/LowerBound": VAL_LB,
                    "val/Likelihood P(x|z)": VAL_PX_Z,
                    "val/KL_z1": VAL_KLD_Z1,
                    "val/KL_z2": VAL_KLD_Z2,
                    "val/KL_z3": VAL_KLD_Z3,
                    "val/Log p(mu2)": VAL_LOG_PMU2,
                    "Epoch": epoch
                })

                print(f"====> Validation set lb: {VAL_LB:.4f}")

                # Save checkpoint with best reconstruction
                if check_best(VAL_PX_Z, best_val_likelihood):
                    best_epoch = epoch
                    best_val_likelihood = VAL_PX_Z
                    self.save_checkpoint(
                        epoch=epoch,
                        val_likelihood=VAL_PX_Z,
                        save_metric='val_likelihood',
                    )

                # Save checkpoint with best lowerbound
                if check_best(VAL_LB, best_val_lb):
                    best_epoch = epoch
                    best_val_lb = VAL_LB
                    self.save_checkpoint(
                        epoch=epoch,
                        val_lower_bound=VAL_LB,
                        save_metric='val_lower_bound',
                    )

                # Save the latest checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    val_lower_bound=VAL_LB,
                    save_metric='latest',
                )

                if check_terminate(epoch, best_epoch, self.config['training_args']['patience'], self.config['training_args']['epochs']):
                    print(f"Training terminated after observing no improvement in {self.config['training_args']['patience']} epochs.")
                    break
   
        print("Training complete!")


def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True,
        help="Path to training config file (json)."
    )
    parser.add_argument(
        "--suf",
        type=str,
        default='',
        help="Experiment specific suffixes",
    )
    parser.add_argument(
        "--parallel", action='store_true', help="Enable single node multi-gpu training"
    )
    parser.add_argument(
        "--resume_pt", type=str, default=None, help="Resume training"
    )
    
    return parser

def main_worker(rank, world_size, args):
    # Initialize Process Group
    world_size, local_rank = ddp_setup()
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    set_seed(config['seed'], rank)

    trainer = Trainer(config, args=args, rank=local_rank, world_size=world_size, parallel=True)
    trainer.train()

    dist.destroy_process_group()

def main(ARGS, rank):
    # Load config file
    with open(ARGS.config, 'r') as f:
        config = json.load(f)
    
    set_seed(config['seed'], rank)
    trainer = Trainer(config, args=ARGS, rank=rank, world_size=1, parallel=False)
    trainer.train()

if __name__ == "__main__":
    parser = args()
    ARGS = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    
    if ARGS.parallel:
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, ARGS))
    else:
        main(ARGS, rank=0)
