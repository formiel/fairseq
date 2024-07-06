import torch


class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
    
    def forward(self, pred, target):
        # pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        print(f"true_dist: {true_dist}")
        return torch.sum(-true_dist * pred, dim=1).sum()
    

num_classes = 5
epsilon = 0.1
eps_i = epsilon / num_classes
x = torch.randn(1, num_classes)
y = torch.randint(num_classes, size=[1])
print(f"x: {x}")
print(f"y: {y}")
loss = label_smooth_loss(num_classes=num_classes, smoothing=0.0)
print(f"Loss without label smoothing: {loss(x, y)}")

nll_loss = -x.gather(dim=-1, index=y.unsqueeze(0))
smooth_loss = -x.sum(dim=-1, keepdim=True)
loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
print(f"nll_loss={nll_loss}, smooth_loss={smooth_loss}, loss={loss}")

loss1 = label_smooth_loss(num_classes=num_classes, smoothing=0.1)
loss2 = torch.nn.functional.cross_entropy(label_smoothing=0.1, reduce="sum")
print(loss1(x,y), loss2(x,y))