from datetime import datetime

import torch
import torch.nn as nn
import pandas as pd

from dataset.complex_as_channel import ispectro, icomplex_as_channel
from settings import HOP_LENGTH, FRAMES_PER_SAMPLE


class EncoderLayer(nn.Module):
    def __init__(self, in_channels=4, out_channels=None, kernel_size=4, stride=2, padding=1, leakiness=0.2, dropout=0,
                 downsample=True, mid_channels=None):
        super().__init__()

        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels * 2
        if mid_channels:
            mid_channels = mid_channels
        else:
            mid_channels = in_channels
        if dropout != 0:
            layers = [nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU(negative_slope=leakiness),
                      nn.Dropout(p=dropout)]
        else:
            layers = [nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU(negative_slope=leakiness)]

        if downsample:
            downsample_layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=False),
                                 nn.BatchNorm2d(mid_channels),
                                 nn.LeakyReLU(negative_slope=leakiness)]

            layers = downsample_layers + layers

        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)


class DecoderLayer(nn.Module):
    def __init__(self, in_channels=4, out_channels=None, kernel_size=4, stride=2, padding=1, leakiness=0.2, dropout=0):
        super().__init__()

        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels // 2

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        if dropout != 0:
            self.sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=leakiness),
                nn.Dropout(p=dropout)
            )
        else:
            self.sequence = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=leakiness)
            )

    def forward(self, x1, x2):
        '''
        x1 is input from decoder path
        x2 is input from skip connection
        '''

        # Upsample via trans conv and half the number of channels
        x1 = self.up_conv(x1)

        # Implement skip connection
        x = torch.cat([x2, x1], dim=1)

        return self.sequence(x)


class UNet(nn.Module):
    def __init__(self, source='vocals', num_layers=5, num_channels=64, kernel_size=4, stride=2, padding=1, dropout=0.5,
                 leakiness=0.2, num_dropout_enc_layers=0, num_dropout_dec_layers=None, normalise_input=True,
                 spec_output=False, timestamp=None):
        super().__init__()
        self.source = source
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.leakiness = leakiness
        self.normalise_input = normalise_input
        self.spec_output = spec_output
        self.num_dropout_enc_layers = num_dropout_enc_layers
        if num_dropout_dec_layers is None:
            self.num_dropout_dec_layers = (num_layers // 2)
        else:
            self.num_dropout_dec_layers = num_dropout_dec_layers

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = timestamp

        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(EncoderLayer(in_channels=4, out_channels=num_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, downsample=False))
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(nn.Conv2d(num_channels, 4, kernel_size=1, stride=1, bias=False))

        for i in range(num_layers - 1):
            if num_layers - 1 - i <= self.num_dropout_enc_layers:
                self.enc_layers.append(EncoderLayer(in_channels=num_channels * 2 ** i, kernel_size=kernel_size,
                                                    stride=stride, padding=padding, leakiness=leakiness,
                                                    dropout=dropout))
            else:
                self.enc_layers.append(EncoderLayer(in_channels=num_channels * 2 ** i, kernel_size=kernel_size,
                                                    stride=stride, padding=padding))

            if num_layers - 1 - i <= self.num_dropout_dec_layers:
                self.dec_layers.append(DecoderLayer(in_channels=num_channels * 2 ** (i + 1), kernel_size=kernel_size,
                                                    stride=stride, padding=padding, dropout=0.5))
            else:
                self.dec_layers.append(DecoderLayer(in_channels=num_channels * 2 ** (i + 1), kernel_size=kernel_size,
                                                    stride=stride, padding=padding))

    def __repr__(self):
        enc_layers_str = ''
        dec_layers_str = ''
        for i in range(self.num_layers):
            enc_layers_str += f'Encoder Block {i}: \n' + repr(self.enc_layers[i]) + '\n\n'
            dec_layers_str += f'Decoder Block {i}: \n' + repr(self.dec_layers[i]) + '\n\n'
        return self.__class__.__name__ + '\n\n' + enc_layers_str + dec_layers_str

    def get_num_params(self):
        total_params = sum(param.numel() for param in self.parameters())
        total_trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        num_params_str = f'Model has a total of {total_params} parameters. \nModel has a total of ' \
                         f'{total_trainable_params} trainable parameters.'
        return num_params_str

    def get_attributes_str(self):
        attribute_str = f'{self.timestamp} MODEL ATTRIBUTES: \nSource to isolate: {self.source} ' \
                        f'\nNumber of channels after the first encoder layer: {self.num_channels} ' \
                        f'\nNumber of hierarchical layers: {self.num_layers} \nKernel size: {self.kernel_size} ' \
                        f'\nStride: {self.stride} \nPadding: {self.padding} \nDropout: {self.dropout} ' \
                        f'\nLeakiness (for ReLU): {self.leakiness} \nNumber of encoder layers with dropout: ' \
                        f'{self.num_dropout_enc_layers} \nNumber of decoder layers with dropout: ' \
                        f'{self.num_dropout_dec_layers} \nInput was normalised: {self.normalise_input} ' \
                        f'\nSpectrogram output: {self.spec_output}'
        return attribute_str

    def forward(self, x):
        # Normalise input
        if self.normalise_input:
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            x = (x - mean) / (1e-6 + std)

        # Create list of encoder path outputs for skip connections
        skip_connections = []

        # Encoder layers
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
            if i < self.num_layers - 1:
                skip_connections.append(x)

        # Decoder layers
        for i in range(self.num_layers - 1, -1, -1):
            dec_layer = self.dec_layers[i]
            if i > 0:
                x = dec_layer(x, skip_connections[i - 1])
                del skip_connections[i - 1]
            else:
                x = dec_layer(x)

        if self.normalise_input:
            x = x * std + mean

        # Formatting output
        if not self.spec_output:
            x = icomplex_as_channel(x)
            x = ispectro(x, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)

        return x
