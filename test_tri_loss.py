import torch


def main():
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)
    import pdb;pdb.set_trace()
    output = triplet_loss(anchor, positive, negative)

if __name__ == "__main__":
    main()
