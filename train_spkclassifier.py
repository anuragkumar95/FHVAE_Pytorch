import os
import wandb
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from models.fhvae import FHVAE
from models.spk_classifier import SpeakerClassifier
from utils import create_output_dir_name
from Datasets.datasets import NumpySpeakerDataset



class SpeakerClassifierTrainer:
    def __init__(self, fhvae_model: FHVAE, spk_classifier: nn.Module, device: str = 'cpu'):
        self.device = torch.device("cuda:0" if device=='gpu' else "cpu")
        self.fhvae_model = fhvae_model.to(self.device)
        self.spk_classifier = spk_classifier.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.spk_classifier.parameters(), lr=0.001)

    def accuracy(self, logits, labels):
        _, preds = torch.max(logits, 1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    def train(self, train_loader, val_loader, test_loader, epochs: int, args):
        self.fhvae_model.eval()  # Freeze FHVAE model
        for epoch in range(epochs):
            self.spk_classifier.train()
            total_loss = 0.0
            total_acc = 0.0
            pbar = tqdm(range(len(train_loader)))
            iterator = iter(train_loader)
            for batch_idx in pbar:
                try:
                    (features, labels) = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    (features, labels) = next(iterator)

                features, labels = features.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    latents = self.fhvae_model.extract_latents(features)
                
                    if args.z1:
                        x = latents['z1']['mu']
                    elif args.z2:
                        x = latents['z2']['mu']
                    else:
                        raise ValueError("Please specify either --z1 or --z2 to select latent features.")

                logits = self.spk_classifier(x.float())
                loss = self.criterion(logits, labels)
                acc = self.accuracy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_acc += acc

                wandb.log({
                    "Step": epoch * len(train_loader) + batch_idx,
                    "Train_Loss": loss.item(),
                    "Train_Acc": acc
                })

                pbar.set_postfix({
                    "Epoch": epoch,
                    "Cur_Loss": loss.item(), 
                    "Avg_Acc": total_acc / ((batch_idx+1)),
                    "Avg_Loss": total_loss / ((batch_idx+1)) 
                })

            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

            wandb.log({
                "Epoch": epoch,
                "Avg_Train_Loss": avg_loss,
                "Avg_Train_Acc": avg_acc
            })

            self.spk_classifier.eval()
            pbar = tqdm(range(len(val_loader)))
            val_iterator = iter(val_loader)
            val_acc = 0.0
            val_loss = 0.0
       
            for batch_idx in pbar:
                try:
                    (features, labels) = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_loader)
                    (features, labels) = next(val_iterator)

                features, labels = features.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    latents = self.fhvae_model.extract_latents(features)
                    if args.z1:
                        x = latents['z1']['mu']
                    elif args.z2:
                        x = latents['z2']['mu']
                    else:
                        raise ValueError("Please specify either --z1 or --z2 to select latent features.")
                    
                    logits = self.spk_classifier(x.float())
                    loss = self.criterion(logits, labels)

                acc = self.accuracy(logits, labels)
                val_acc += acc
                val_loss += loss.item()

                pbar.set_postfix({
                    "Epoch": epoch,
                    "Cur_Val_Loss": loss.item(), 
                    "Avg_Val_Loss": val_loss / ((batch_idx+1)),
                    "Avg_Val_Acc": val_acc / ((batch_idx+1))
                })

            val_avg_loss = val_loss / len(val_loader)
            val_avg_acc = val_acc / len(val_loader)
            print(f"Validation | Loss: {val_avg_loss:.4f} | Acc: {val_avg_acc:.4f}")

            wandb.log({
                "Epoch": epoch,
                "Avg_Val_Loss": val_avg_loss,
                "Avg_Val_Acc": val_avg_acc
            })

        # Run on testset finally
        pbar = tqdm(range(len(test_loader)))
        test_iterator = iter(test_loader)
        test_acc = 0.0
        test_loss = 0.0
        for batch_idx in pbar:
            try:
                (features, labels) = next(test_iterator)
            except StopIteration:
                test_iterator = iter(test_loader)
                (features, labels) = next(test_iterator)

            features, labels = features.to(self.device), labels.to(self.device)

            with torch.no_grad():
                latents = self.fhvae_model.extract_latents(features)
                if args.z1:
                    x = latents['z1']['mu']
                elif args.z2:
                    x = latents['z2']['mu']
                else:
                    raise ValueError("Please specify either --z1 or --z2 to select latent features.")
                    
                logits = self.spk_classifier(x.float())
                loss = self.criterion(logits, labels)

            acc = self.accuracy(logits, labels)
            test_acc += acc
            test_loss += loss.item()

        test_acc_avg = test_acc / len(test_loader)
        test_loss_avg = test_loss / len(test_loader)
        print(f"Testset | Loss: {test_loss_avg:.4f} | Acc: {test_acc_avg:.4f}")


def main(ARGS):
    wandb.login()
    wandb.init(project='FHVAE_spk_classification', name=f"{'z1' if ARGS.z1 else 'z2'}_latents")

    # Load Dataset
    dataset = "timit"
    feat_type = "spec"
    data_format = "numpy"
    min_len = 20
    seg_len = 20
    seg_shift = 8
    mvn_path = "./misc/mvn.json"
    rand_seg = False
    dataset_dir = create_output_dir_name(
            dataset, data_format, feat_type
        )
    train_feat_scp = dataset_dir / "train" / "feats.scp"
    train_len_scp = dataset_dir / "train" / "len.scp"
    train_dataset_args = [
        train_feat_scp,
        train_len_scp,
        min_len,
        mvn_path,
        seg_len,
        seg_shift,
        rand_seg,
    ]

    train_dataset = NumpySpeakerDataset(*train_dataset_args, split="train")
    dev_dataset = NumpySpeakerDataset(*train_dataset_args, split="val")
    test_dataset = NumpySpeakerDataset(*train_dataset_args, split="test")

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
        )
    
    val_loader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=4
    )

    # Load pre-trained FHVAE model
    fhvae_model = FHVAE(
        input_size=201,
        z1_hus=256,
        z2_hus=256,
        z1_dim=32,
        z2_dim=32,
        x_hus=256
    )
    exp_dir = f"/users/PAS2301/kumar1109/PyTorch-ScalableFHVAE/experiments/fhvae_timit_e500_b256_a10.0_z32_h256_LSTM1_"
    ckpt = torch.load(f"{exp_dir}/best_checkpoint_best_val_lower_bound.pt", weights_only=False)
    fhvae_model.load_state_dict(ckpt['model_state_dict'])
    fhvae_model = fhvae_model.double()

    #Create Speaker Classifier
    num_speakers = len(train_dataset.spk2idx)
    print(  f"Number of speakers: {num_speakers}")
    spk_classifier = SpeakerClassifier(input_size=32, num_speakers=num_speakers, hus=256)

    trainer = SpeakerClassifierTrainer(
        fhvae_model, spk_classifier, device='cuda'
    )

    trainer.train(train_loader, val_loader, test_loader, 500, ARGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z1", action='store_true', help="Use z1 latents")
    parser.add_argument("--z2", action='store_true', help="Use z2 latents")
    args = parser.parse_args()
    main(args)