import torch
import torch.nn as nn

## I told gpt to write this for me
class ModifiedEuclideanLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target):
        distance = torch.norm(pred - target, dim=1)  # Calculate Euclidean distance
        
        # Apply a margin to allow for dissimilarity
        loss = torch.mean(torch.relu(distance - self.margin))
        
        return loss
