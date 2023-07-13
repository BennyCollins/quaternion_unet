import math

import torch
import torch.nn as nn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def channel_split_quaternion(input):
    quaternion_list = []
    input_rijk = input.chunk(4, dim=1)
    batches, num_quaternions, h, w = input_rijk[0].shape
    for quaternion_index in torch.range(0, num_quaternions-1, dtype=torch.int64, device=device):
        for component_index in range(len(input_rijk)):
            component = input_rijk[component_index]
            component_n = torch.index_select(component, 1, torch.tensor(quaternion_index))
            if component_index == 0:
                quaternion = component_n
            else:
                quaternion = torch.cat((quaternion, component_n), 1)
        quaternion_list.append(quaternion)
    return quaternion_list


def quaternion_max_pool2d(quaternion, pool_size=(2,2), pool_stride=(2,2)):
    '''Implementing quaternion 2D max pooling with dilation=1 and no padding'''
    batches, channels, h, w = quaternion.shape
    h_new = math.floor((h - pool_size[0])/pool_stride[0] + 1)
    w_new = math.floor((w - pool_size[1])/pool_stride[1] + 1)
    amp = torch.norm(quaternion, dim=1, keepdim=True)
    amp_uf = amp.unfold(2, pool_size[0], pool_stride[0]).unfold(3, pool_size[1], pool_stride[1]).reshape(batches, channels//4, h_new, w_new, pool_size[0] * pool_size[1])
    qs_uf = quaternion.unfold(2, pool_size[0], pool_stride[0]).unfold(3, pool_size[1], pool_stride[1]).reshape(batches, channels, h_new, w_new, pool_size[0] * pool_size[1])
    maxamp = amp_uf.argmax(-1, keepdim=True).expand(-1, channels, -1, -1, -1)
    return qs_uf.gather(-1, maxamp).view(batches, channels, h_new, w_new)


class QuaternionMaxPool2D(torch.nn.Module):
    def __init__(self, pool_size=2, pool_stride=None):

        super(QuaternionMaxPool2D, self).__init__()

        self.pool_size = nn.modules.utils._pair(pool_size)
        if pool_stride is None:
            self.pool_stride = nn.modules.utils._pair(pool_size)
        else:
            self.pool_stride = nn.modules.utils._pair(pool_stride)

    def forward(self, input):
        assert input.size(dim=1) % 4 == 0, 'Tensor dimension should be divisible by number of chunks'
        input_quaternion_list = channel_split_quaternion(input)
        for n in range(len(input_quaternion_list)):
            pooled_quaternion = quaternion_max_pool2d(input_quaternion_list[n], self.pool_size, self.pool_stride)
            r_n, i_n, j_n, k_n = pooled_quaternion.chunk(4, dim=1)
            if n == 0:
                r, i, j, k = r_n, i_n, j_n, k_n
            else:
                r = torch.cat((r, r_n), 1)
                i = torch.cat((i, i_n), 1)
                j = torch.cat((j, j_n), 1)
                k = torch.cat((k, k_n), 1)
        output = torch.cat((r, i, j, k), 1)
        return output