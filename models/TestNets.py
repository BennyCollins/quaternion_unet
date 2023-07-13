from datetime import datetime

import torch.nn as nn
import pandas as pd

from dataset.complex_as_channel import ispectro, icomplex_as_channel
from quaternion_components.quaternion_layers import QuaternionConv, QuaternionTransposeConv
from quaternion_components.quaternion_batch_norm import QuaternionBatchNorm2d
from settings import HOP_LENGTH, FRAMES_PER_SAMPLE


class TestNet(nn.Module):
    def __init__(self, num_layers=10, num_channels=48, kernel_size=4, stride=2, padding=1,
                 source='vocals', spec_output=False, timestamp=None):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.source = source
        self.spec_output = spec_output

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = timestamp

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(4, num_channels, kernel_size, stride, padding))
        self.layers.append(nn.BatchNorm2d(num_channels))
        self.layers.append(nn.ConvTranspose2d(num_channels, 4, kernel_size, stride, padding))
        self.layers.append(nn.BatchNorm2d(4))

    def __repr__(self):
        return repr(self.layers)

    def get_attributes_str(self):
        attribute_str = f'num_channels: {self.num_channels} \nnum_layers: {self.num_layers} \nkernel_size: ' \
                        f'{self.kernel_size} \nstride: {self.stride} \npadding: {self.padding} \nsource: {self.source} ' \
                        f'\nnormalise_input: {self.normalise_input} \nspec_output: {self.spec_output}'
        return attribute_str

    def get_num_params(self):
        total_params = sum(param.numel() for param in self.parameters())
        total_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        num_params_str = f'Model has a total of {total_params} parameters. \nModel has a total of ' \
                         f'{total_trainable_params} trainable parameters.'
        return num_params_str

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if not self.spec_output:
            x = icomplex_as_channel(x)
            x = ispectro(x, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)

        return x


class QTestNet(nn.Module):
    def __init__(self, num_layers=10, num_channels=48, kernel_size=4, stride=2, padding=1, source='vocals',
                 spec_output=False, timestamp=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.source = source
        self.spec_output = spec_output

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = timestamp

        self.layers = nn.ModuleList()
        self.layers.append(QuaternionConv(4, num_channels, kernel_size, stride, padding))
        self.layers.append(QuaternionBatchNorm2d(num_channels))
        self.layers.append(QuaternionTransposeConv(num_channels, 4, kernel_size, stride, padding))
        self.layers.append(QuaternionBatchNorm2d(4))

    def __repr__(self):
        return repr(self.layers)

    def get_attributes_str(self):
        attribute_str = f'num_channels: {self.num_channels} \nnum_layers: {self.num_layers} \nkernel_size: ' \
                        f'{self.kernel_size} \nstride: {self.stride} \npadding: {self.padding} \nsource: {self.source} ' \
                        f'\nnormalise_input: {self.normalise_input} \nspec_output: {self.spec_output}'
        return attribute_str

    def get_num_params(self):
        total_params = sum(param.numel() for param in self.parameters())
        total_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        num_params_str = f'Model has a total of {total_params} parameters. \nModel has a total of ' \
                         f'{total_trainable_params} trainable parameters.'
        return num_params_str

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        if not self.spec_output:
            x = icomplex_as_channel(x)
            x = ispectro(x, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)

        return x