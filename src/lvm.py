
import torch
import torch.nn as nn

class LVM(nn.Module):
    def __init__(self, params):
        super(LVM, self).__init__()

        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.lvm_dim = params.lvm_dim

        self.mean = nn.Linear(self.input_dim, self.lvm_dim)
        self.var = nn.Linear(self.input_dim, self.lvm_dim)

    def forward(self, encoded):
        mean = self.mean(encoded)
        var = self.var(encoded)

        if self.training == True:
            # generate noise
            size = var.size()
            noise = torch.randn(size)
            noise = noise.cuda()
            # produce var
            z = mean + torch.exp(var / 2) * noise * 0.5
        else:
            # in the inference time, we only use the true mean
            z = mean
        
        return z