import torch.nn as nn
import torch
import torch.nn.functional as F


class SpeakerClassifier(nn.Module):
    """Speaker classifier network

    Args:
        input_size: Size of input features
        num_speakers: Number of speakers to classify
        hus:        Number of hidden units in FC layers

    Returns:
        out: Logits for speaker classification

    """

    def __init__(self, input_size: int, num_speakers: int, hus: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hus)
        self.fc2 = nn.Linear(hus, num_speakers)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        return out