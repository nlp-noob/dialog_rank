import torch
import torch.nn as nn
import torch.nn.functional as F

# define model with forward method

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        similarity_positive = F.cosine_similarity(anchor, positive, dim=1, eps=1e-8)
        similarity_negative = F.cosine_similarity(anchor, negative, dim=1, eps=1e-8)
        distance_positive = torch.acos(similarity_positive)
        distance_negative = torch.acos(similarity_negative)
        loss = torch.mean(torch.relu(distance_positive - distance_negative + self.margin))
        return loss


def main():
    # define dummy input
    torch.manual_seed(12834)
    anchor = torch.randn(32, 128)
    positive = torch.randn(32, 128)
    negative = torch.randn(32, 128)

    anchor = F.normalize(anchor)
    positive = F.normalize(positive)
    negative = F.normalize(negative)

    criterion = TripletLoss()
    loss = criterion(anchor, positive, negative)
    
    print(f"Anchor: {anchor}")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    print(f"Triplet Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
