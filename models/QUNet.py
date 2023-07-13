from datetime import datetime
import time

import torch
import torch.nn as nn
import pandas as pd

from dataset.complex_as_channel import ispectro, icomplex_as_channel
from quaternion_components.quaternion_layers import QuaternionConv, QuaternionTransposeConv
from quaternion_components.quaternion_batch_norm import QuaternionBatchNorm2d
from quaternion_components.quaternion_dropout import QuaternionDropout
from settings import HOP_LENGTH, FRAMES_PER_SAMPLE


class QEncoderLayer(nn.Module):
    def __init__(self, in_channels=4, out_channels=None, kernel_size=4, stride=2, padding=1, leakiness=0.2, dropout=0,
                 downsample=True, mid_channels=None, quaternion_dropout=True, quaternion_norm=True):
        super().__init__()

        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels * 2
        if mid_channels:
            mid_channels = mid_channels
        else:
            mid_channels = in_channels

        layers = [QuaternionConv(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)]

        if quaternion_norm:
            layers.append(QuaternionBatchNorm2d(out_channels))
        else:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=leakiness))

        if dropout != 0:
            if quaternion_dropout:
                layers.append(QuaternionDropout(p=dropout))
            else:
                layers.append(nn.Dropout(p=dropout))

        if downsample:
            downsample_layers = [QuaternionConv(in_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=False)]
            if quaternion_norm:
                downsample_layers.append(QuaternionBatchNorm2d(mid_channels))
            else:
                downsample_layers.append(nn.BatchNorm2d(mid_channels))

            downsample_layers.append(nn.LeakyReLU(negative_slope=leakiness))

            layers = downsample_layers + layers

        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)


class QDecoderLayer(nn.Module):
    def __init__(self, in_channels=4, out_channels=None, kernel_size=4, stride=2, padding=1, leakiness=0.2, dropout=0,
                 quaternion_dropout=True, quaternion_norm=True):
        super().__init__()

        if out_channels:
            out_channels = out_channels
        else:
            out_channels = in_channels // 2

        up_conv_layers = [QuaternionTransposeConv(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride,
                                                  padding=padding, bias=False)]

        if quaternion_norm:
            up_conv_layers.append(QuaternionBatchNorm2d(in_channels // 2))
        else:
            up_conv_layers.append(nn.BatchNorm2d(in_channels // 2))

        up_conv_layers.append(nn.LeakyReLU(negative_slope=leakiness))

        self.up_conv = nn.Sequential(*up_conv_layers)

        layers = [QuaternionConv(in_channels, out_channels, kernel_size=1, stride=1, bias=False)]

        if quaternion_norm:
            layers.append(QuaternionBatchNorm2d(out_channels))
        else:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope=leakiness))

        if dropout != 0:
            if quaternion_dropout:
                layers.append(QuaternionDropout(p=dropout))
            else:
                layers.append(nn.Dropout(p=dropout))

        self.sequence = nn.Sequential(*layers)

    def forward(self, x1, x2):
        '''
        x is input from decoder path
        x2 is input from skip connection
        '''

        # Upsample via trans conv and half the number of channels
        x = self.up_conv(x1)

        # Implement skip connection
        x = torch.cat([x2, x], dim=1)

        return self.sequence(x)


class DefaultQUNet(nn.Module):
    def __init__(self, num_layers=4, num_channels=64, kernel_size=4, stride=2, padding=1, source='vocals',
                 quaternion_dropout=True, quaternion_batch_norm=True, spec_output=False, timestamp=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.source = source
        self.spec_output = spec_output

        if timestamp is None:
            self.timestamp = 'Q_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = 'Q_' + timestamp

        self.down1 = QEncoderLayer(in_channels=4, out_channels=num_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, downsample=False)
        self.down2 = QEncoderLayer(in_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.down3 = QEncoderLayer(in_channels=num_channels * 2, kernel_size=kernel_size, stride=stride,
                                   padding=padding)
        self.down4 = QEncoderLayer(in_channels=num_channels * 4, kernel_size=kernel_size, stride=stride,
                                   padding=padding)
        self.down5 = QEncoderLayer(in_channels=num_channels * 8, kernel_size=kernel_size, stride=stride,
                                   padding=padding)

        self.up1 = QDecoderLayer(in_channels=num_channels * 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.up2 = QDecoderLayer(in_channels=num_channels * 8, kernel_size=kernel_size, stride=stride, padding=padding)
        self.up3 = QDecoderLayer(in_channels=num_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.up4 = QDecoderLayer(in_channels=num_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out_conv = QuaternionTransposeConv(num_channels, 4, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # Encoder path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.down5(x4)

        # Decoder path
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.out_conv(x)

        if not self.spec_output:
            output = icomplex_as_channel(output)
            output = ispectro(output, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)

        return output


class QUNet(nn.Module):
    def __init__(self, source='vocals', num_layers=5, num_channels=64, kernel_size=4, stride=2, padding=1, dropout=0.5,
                 leakiness=0.2, quaternion_dropout=True, quaternion_norm=True, num_dropout_enc_layers=0,
                 num_dropout_dec_layers=None, normalise_input=True, spec_output=False, timestamp=None):
        super().__init__()
        self.source = source
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.leakiness = leakiness
        self.quaternion_dropout = quaternion_dropout
        self.quaternion_norm = quaternion_norm
        self.normalise_input = normalise_input
        self.spec_output = spec_output
        self.num_dropout_enc_layers = num_dropout_enc_layers
        if num_dropout_dec_layers is None:
            self.num_dropout_dec_layers = (num_layers // 2)
        else:
            self.num_dropout_dec_layers = num_dropout_dec_layers

        if timestamp is None:
            self.timestamp = 'Q_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.timestamp = 'Q_' + timestamp

        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(QEncoderLayer(in_channels=4, out_channels=num_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding, leakiness=leakiness, downsample=False))
        self.dec_layers = nn.ModuleList()
        self.dec_layers.append(QuaternionConv(num_channels, 4, kernel_size=1, stride=1, bias=False))

        for i in range(num_layers - 1):
            if num_layers - 1 - i <= self.num_dropout_enc_layers:
                self.enc_layers.append(QEncoderLayer(in_channels=num_channels * 2 ** i, kernel_size=kernel_size,
                                                     stride=stride, padding=padding, leakiness=leakiness,
                                                     dropout=dropout, quaternion_dropout=quaternion_dropout,
                                                     quaternion_norm=quaternion_norm))
            else:
                self.enc_layers.append(QEncoderLayer(in_channels=num_channels * 2 ** i, kernel_size=kernel_size,
                                                     stride=stride, padding=padding, leakiness=leakiness,
                                                     quaternion_norm=quaternion_norm))

            if num_layers - 1 - i <= self.num_dropout_dec_layers:
                self.dec_layers.append(QDecoderLayer(in_channels=num_channels * 2 ** (i + 1), kernel_size=kernel_size,
                                                     stride=stride, padding=padding, leakiness=leakiness,
                                                     dropout=dropout, quaternion_dropout=quaternion_dropout,
                                                     quaternion_norm=quaternion_norm))
            else:
                self.dec_layers.append(QDecoderLayer(in_channels=num_channels * 2 ** (i + 1), kernel_size=kernel_size,
                                                     stride=stride, padding=padding, leakiness=leakiness,
                                                     quaternion_norm=quaternion_norm))

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
                        f'{self.num_dropout_dec_layers} \nQuaternion dropout used: {self.quaternion_dropout} ' \
                        f'\nQuaternion normalisation used: {self.quaternion_norm} ' \
                        f'\nInput was normalised: {self.normalise_input} \nSpectrogram output: {self.spec_output}'
        return attribute_str

    def normalise_quaternion(self, x):
        [r, i, j, k] = torch.chunk(x, 4, dim=1)
        r_mean = torch.mean(r).item()
        i_mean = torch.mean(i).item()
        j_mean = torch.mean(j).item()
        k_mean = torch.mean(k).item()
        r, i, j, k = r - r_mean, i - i_mean, j - j_mean, \
                     k - k_mean

        quat_variance = torch.mean(r ** 2 + i ** 2 + j ** 2 + k ** 2).data
        denominator = torch.sqrt(quat_variance.item() + torch.tensor(1e-5))

        # Normalize
        r = r / denominator
        i = i / denominator
        j = j / denominator
        k = k / denominator
        return torch.cat((r, i, j, k), dim=1), r_mean, i_mean, j_mean, k_mean, denominator

    def inverse_normalise_quaternion(self, x, r_mean, i_mean, j_mean, k_mean, denominator):
        x = x * denominator
        [r, i, j, k] = torch.chunk(x, 4, dim=1)
        r, i, j, k = r + r_mean, i + i_mean, j + j_mean, k + k_mean
        return torch.cat((r, i, j, k), dim=1)

    def forward(self, x):
        # Normalise input
        if self.normalise_input:
            if self.quaternion_norm:
                x, r_mean, i_mean, j_mean, k_mean, denominator = self.normalise_quaternion(x)
            else:
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                std = x.std(dim=(1, 2, 3), keepdim=True)
                x = (x - mean) / (1e-6 + std)

        # Create list of encoder path outputs for skip connections
        skip_connections = {}

        # Encoder layers
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x)
            if i < self.num_layers - 1:
                skip_connections[i] = x

        # Decoder layers
        for i in range(self.num_layers - 1, -1, -1):
            dec_layer = self.dec_layers[i]
            if i > 0:
                x = dec_layer(x, skip_connections[i - 1])
            else:
                x = dec_layer(x)

        if self.normalise_input:
            if self.quaternion_norm:
                x = self.inverse_normalise_quaternion(x, r_mean, i_mean, j_mean, k_mean, denominator)
            else:
                x = x * std + mean

        # Formatting output
        if not self.spec_output:
            x = icomplex_as_channel(x)
            x = ispectro(x, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)

        return x
