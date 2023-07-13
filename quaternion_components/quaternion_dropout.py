import torch
import torch.nn as nn


class QuaternionDropout(nn.Module):
    def __init__(self, p=0.5):
        super(QuaternionDropout, self).__init__()
        self.p = p

    def __repr__(self):
        return f'QuaternionDropout(p={self.p})'

    def forward(self, x):
        if self.training and self.p != 0:
            mask_size = [x.size(0), int(x.size(1) // 4), x.size(2), x.size(3)]
            condition_tensor = torch.rand(mask_size)
            condition_tensor = condition_tensor > self.p
            condition_tensor = condition_tensor.long()
            condition_tensor = \
                torch.cat([condition_tensor, condition_tensor, condition_tensor, condition_tensor], dim=1).to(x.device)
            x = x * condition_tensor
        return x
