import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, util

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
    # torch.manual_seed(12834)
    # anchor = torch.randn(32, 128)
    # positive = torch.randn(32, 128)
    # negative = torch.randn(32, 128)


    anchor_s = "You both will be reunite" 
    positive_s = "You both will reunite",
    negative_s = "u will meet someone speicail soon"
    model = SentenceTransformer("all-MiniLM-L6-v2")

    anchor = model.encode(anchor_s)
    positive = model.encode(positive_s)
    negative = model.encode(negative_s)

    p_score = util.cos_sim(anchor, positive).tolist()[0][0]
    n_score = util.cos_sim(anchor, negative).tolist()[0][0]

    anchor = F.normalize(torch.tensor(anchor))
    positive = F.normalize(torch.tensor(positive))
    negative = F.normalize(torch.tensor(negative))

    criterion = TripletLoss()
    loss = criterion(anchor, positive, negative)
    
    # print(f"Anchor: {anchor}")
    # print(f"Positive: {positive}")
    # print(f"Negative: {negative}")
    print(f"P score: {p_score.item()}")
    print(f"N score: {n_score.item()}")
    print(f"Triplet Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
