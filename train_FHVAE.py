import sys
import argparse
import os
import json
import random
from tqdm import tqdm
import torch
from pathlib import Path
from models.fhvae import FHVAE
from torch.optim import Adam
import numpy as np
from Datasets.datasets_eeg import NumpyEEGDataset
from Datasets.datasets import NumpyDataset
from utils import (
    save_checkpoint,
    check_best,
)
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import contextlib
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
    return -1 * torch.mean(lower_bound) + alpha * log_qy


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
            self.train_ds = NumpyEEGDataset(
                **config['data_args'], split='train'
            )
            self.val_ds = NumpyEEGDataset(
                **config['data_args'], split='val'
            )
            self.test_ds = NumpyEEGDataset(
                **config['data_args'], split='test'
            )
            
        if config['task'] == 'timit':
            ds_dir = config['dataset_dir']
            self.train_ds = NumpyDataset(
                **config['data_args'], 
                feat_scp=f"{ds_dir}/train/feats.scp",
                len_scp=f"{ds_dir}/train/len.scp",
                split='train'
            )
            self.val_ds = NumpyDataset(
                **config['data_args'],
                feat_scp=f"{ds_dir}/dev/feats.scp",
                len_scp=f"{ds_dir}/dev/len.scp",
                split='val'
            )
            self.test_ds = NumpyDataset(
                **config['data_args'], 
                feat_scp=f"{ds_dir}/test/feats.scp",
                len_scp=f"{ds_dir}/test/len.scp",
                split='test'
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
        self.model = FHVAE(**config['model_args']).to(self.device).double()
        if parallel and rank is not None:
            self.model = DDP(
                self.model, device_ids=[rank], output_device=rank
            )
        
        self.optimizer = Adam(
            self.model.parameters(), 
            **config['optimizer_args']
        )

        if args.resume_pt is not None:
            self.load_checkpoint(args.resume_pt, map_location=self.device)

        self.accum_grad = self.config['training_args']['accum_grad']

        # Logging
        if self.rank == 0:
            print("Train dataloader length:", len(self.train_loader))
            print("Valid dataloader length:", len(self.val_loader))
            print("Test dataloader length:", len(self.test_loader))
            wandb.login()
            wandb.init(project='FHVAE', name=config['run_name'])
            self.exp_dir = f"./experiments/{config['run_name']}_{args.suf}"
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
        print(f"Checkpoint saved at {save_path}")

    def load_checkpoint(self, checkpoint_path, map_location='cpu'):
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

            pbar = tqdm(range(min(self.config['training_args']['steps_per_epoch'], len(self.train_loader)//self.accum_grad)))   
            
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
                        (idxs, features, nsegs) = next(iterator)
                    except StopIteration:
                        iterator = iter(self.train_loader)
                        (idxs, features, nsegs) = next(iterator)

                    features = features.to(self.device)
                    idxs = idxs.to(self.device)
                    nsegs = nsegs.to(self.device)

                    lower_bound, discrim_loss, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_pred = self.model(
                        x=features, mu_idx=idxs, num_seqs=self.train_ds.num_seqs, num_segs=nsegs
                    )
                    loss = loss_function(lower_bound, discrim_loss, self.config['training_args']['alpha_dis']) / self.accum_grad
                    mse = ((x_pred - features)**2).mean().item()
                    loss.backward()
                
                    batch_lb += lower_bound.mean().item() / self.accum_grad
                    batch_disc_loss += discrim_loss.mean().item() / self.accum_grad
                    batch_log_px_z += log_px_z.mean().item() / self.accum_grad
                    batch_neg_kld_z1 += neg_kld_z1.mean().item() / self.accum_grad
                    batch_neg_kld_z2 += neg_kld_z2.mean().item() / self.accum_grad
                    batch_log_pmu2 += log_pmu2.mean().item() / self.accum_grad
                    batch_mse += mse / self.accum_grad
                    batch_loss += loss.item() / self.accum_grad

                self.optimizer.step()
                train_loss += batch_loss

                if torch.isnan(lower_bound).any():
                    print("Training diverged")
                    raise sys.exit(2)

                if self.rank == 0:
                    wandb.log({
                        "Loss": batch_loss / features.shape[1],
                        "MSE": mse,
                        "LowerBound": batch_lb / features.shape[1],
                        "Discrim_Loss": batch_disc_loss / features.shape[1],
                        "Step": epoch * len(self.train_loader) + (batch_idx + 1),
                        "Likelihood P(x|z)": batch_log_px_z / features.shape[1],
                        "KL(q(z1|x,z2)||p(z1))": (-batch_neg_kld_z1) / features.shape[1],
                        "KL(q(z2|x)||p(z2|mu2))": (-batch_neg_kld_z2) / features.shape[1],
                        "Log p(mu2)": batch_log_pmu2 / features.shape[1],
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
            VAL_LOG_PMU2 = 0
            with torch.no_grad():
                pbar = tqdm(self.val_loader)
                batch_idx = 1
                for (idxs, feature, nsegs) in pbar:
                    
                    feature = feature.to(self.device)
                    idxs = idxs.to(self.device)
                    nsegs = nsegs.to(self.device)

                    feature = feature.squeeze(0)
                    if self.config['task'] == 'eeg':
                        feature = feature.permute(2,0,1)

                    val_lower_bound, _, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_pred = self.model(
                        feature, idxs, self.val_ds.num_seqs, nsegs, mode='val'
                    )
                    mse = ((x_pred - feature)**2).mean().item()

                    VAL_LB += val_lower_bound.mean().item()/feature.shape[1]
                    VAL_MSE += mse
                    VAL_PX_Z += log_px_z.mean().item()/feature.shape[1]
                    VAL_KLD_Z1 += (-neg_kld_z1).mean().item()/feature.shape[1] 
                    VAL_KLD_Z2 += (-neg_kld_z2).mean().item()/feature.shape[1]
                    VAL_LOG_PMU2 += log_pmu2.mean().item()/feature.shape[1]
                    
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
            VAL_LOG_PMU2 /= len(self.val_loader)

            if self.rank == 0:
                wandb.log({
                    "Val_MSE": VAL_MSE,
                    "Val_LowerBound": VAL_LB,
                    "Val_Likelihood P(x|z)": VAL_PX_Z,
                    "Val_KL(q(z1|x,z2)||p(z1))": VAL_KLD_Z1,
                    "Val_KL(q(z2|x)||p(z2|mu2))": VAL_KLD_Z2,
                    "Val_Log p(mu2)": VAL_LOG_PMU2,
                    "Epoch": epoch
                })

                print(f"====> Validation set lb: {VAL_LB:.4f}")

            # Test
            TEST_LB = 0
            TEST_MSE = 0
            TEST_PX_Z = 0
            TEST_KLD_Z1 = 0
            TEST_KLD_Z2 = 0
            TEST_LOG_PMU2 = 0
            with torch.no_grad():
                pbar = tqdm(self.test_loader)
                batch_idx = 1
                for (idxs, feature, nsegs) in pbar:

                    feature = feature.to(self.device)
                    idxs = idxs.to(self.device)
                    nsegs = nsegs.to(self.device)

                    feature = feature.squeeze(0)
                    if self.config['task'] == 'eeg':
                        feature = feature.permute(2,0,1)
                
                    test_lower_bound, _, log_px_z, neg_kld_z1, neg_kld_z2, log_pmu2, x_pred = self.model(
                        feature, idxs, self.test_ds.num_seqs, nsegs, mode='test'
                    )
                    
                    mse = ((x_pred - feature)**2).mean().item()

                    TEST_LB += test_lower_bound.mean().item()/feature.shape[1]
                    TEST_MSE += mse
                    TEST_PX_Z += log_px_z.mean().item()/feature.shape[1]
                    TEST_KLD_Z1 += (-neg_kld_z1).mean().item()/feature.shape[1] 
                    TEST_KLD_Z2 += (-neg_kld_z2).mean().item()/feature.shape[1]
                    TEST_LOG_PMU2 += log_pmu2.mean().item()/feature.shape[1]
                    
                    pbar.set_postfix({
                        "Epoch": epoch,
                        "Test_LB": TEST_LB / ((batch_idx+1) * feature.shape[1]) 
                    })
                    batch_idx += 1

            if self.parallel:
                self.reduce_tensor(TEST_LB, self.world_size)
                self.reduce_tensor(TEST_MSE, self.world_size)
                self.reduce_tensor(TEST_PX_Z, self.world_size)
                self.reduce_tensor(TEST_KLD_Z1, self.world_size)
                self.reduce_tensor(TEST_KLD_Z2, self.world_size)
                self.reduce_tensor(TEST_LOG_PMU2, self.world_size)

            TEST_LB /= len(self.test_loader)
            TEST_MSE /= len(self.test_loader)
            TEST_PX_Z /= len(self.test_loader)
            TEST_KLD_Z1 /= len(self.test_loader)
            TEST_KLD_Z2 /= len(self.test_loader)
            TEST_LOG_PMU2 /= len(self.test_loader)

            if self.rank == 0:
                wandb.log({
                    "Test_MSE": TEST_MSE,
                    "Test_LowerBound": TEST_LB,
                    "Test_Likelihood P(x|z)": TEST_PX_Z,
                    "Test_KL(q(z1|x,z2)||p(z1))": TEST_KLD_Z1,
                    "Test_KL(q(z2|x)||p(z2|mu2))": TEST_KLD_Z2,
                    "Test_Log p(mu2)": TEST_LOG_PMU2,
                    "Epoch": epoch
                })

                print(f"====> Test set lb: {TEST_LB:.4f}")

                if check_best(VAL_PX_Z, best_val_likelihood):
                    best_epoch = epoch
                    best_val_likelihood = VAL_PX_Z
                    self.save_checkpoint(
                        epoch=epoch,
                        val_likelihood=VAL_PX_Z,
                        save_metric='val_likelihood',
                    )

                if check_best(VAL_LB, best_val_lb):
                    best_epoch = epoch
                    best_val_lb = VAL_LB
                    self.save_checkpoint(
                        epoch=epoch,
                        val_lower_bound=VAL_LB,
                        save_metric='val_lower_bound',
                    )

                if check_terminate(epoch, best_epoch, self.config['training_args']['patience'], self.config['training_args']['epochs']):
                    print("Training terminated!")
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
