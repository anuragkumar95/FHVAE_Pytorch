import torch
import torch.nn as nn
import torch.nn.functional as F

    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, learnable=True):
        super(InfoNCELoss, self).__init__()
        init_temp = torch.tensor(temperature)
        if learnable:
            self.temperature = nn.Parameter(torch.log(init_temp))
        else:
            self.register_buffer('temperature', torch.log(init_temp))

    def forward(self, z2_eeg, z2_joint, labels=None):
        z2_eeg = F.normalize(z2_eeg, dim=-1)
        z2_joint = F.normalize(z2_joint, dim=-1)
        B = z2_eeg.shape[0]
        temp = torch.exp(self.temperature)
        
        logits = torch.matmul(z2_eeg, z2_joint.T) / temp

        if labels is None:
            # SELF-SUPERVISED: SYMMETRIC 
            target_labels = torch.arange(B).to(z2_eeg.device)
            loss_i = F.cross_entropy(logits, target_labels)
            loss_j = F.cross_entropy(logits.T, target_labels)
            return (loss_i + loss_j) / 2
        
        else:
            # SUPERVISED: SYMMETRIC WITH MASK 
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(z2_eeg.device)

            # Zeroing out the diagonal to avoid trivial solutions where a sample is most similar to itself
            diag_mask = 1 - torch.eye(B).to(z2_eeg.device)
            mask = mask * diag_mask
            
            # EEG-to-Joint Loss
            mask_i = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
            log_probs_i = F.log_softmax(logits, dim=1)
            loss_i = -(mask_i * log_probs_i).sum(dim=1).mean()

            # 2. Joint-to-EEG Loss
            mask_j = mask / (mask.sum(dim=0, keepdim=True) + 1e-8)
            log_probs_j = F.log_softmax(logits.T, dim=1)
            loss_j = -(mask_j * log_probs_j).sum(dim=1).mean()

            return (loss_i + loss_j) / 2

