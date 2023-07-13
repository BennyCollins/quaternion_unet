import gc

import torch
from torch.nn import Module, Parameter


'''
Code taken from url below with minor restructuring to save memory:
repo url: https://github.com/eleGAN23/QGAN
file url: https://github.com/eleGAN23/QGAN/blob/main/utils/QBN_Vecchi2.py

@article{grassucci2021quaternion,
      title={Quaternion Generative Adversarial Networks},
      author={Grassucci, Eleonora and Cicero, Edoardo and Comminiello, Danilo},
      year={2021},
      eprint={2104.09630},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''


def moving_average_update(statistic, curr_value, momentum):
    new_value = (1 - momentum) * statistic + momentum * curr_value

    return new_value.data


class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)

        self.register_buffer('moving_var', torch.ones(1))
        self.register_buffer('moving_mean', torch.zeros(4))
        self.momentum = momentum

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def normalise_quaternion(self, x):
        [r, i, j, k] = torch.chunk(x, 4, dim=1)
        mu = torch.stack([torch.mean(r), torch.mean(i), torch.mean(j), torch.mean(k)], dim=0).data
        r, i, j, k = r - mu[0].item(), i - mu[1].item(), j - mu[2].item(), k - mu[3].item()

        quat_variance = torch.mean(r ** 2 + i ** 2 + j ** 2 + k ** 2).data
        denominator = torch.sqrt(quat_variance.item() + self.eps)

        # Normalize
        r = r / denominator
        i = i / denominator
        j = j / denominator
        k = k / denominator
        return r, i, j, k, mu, quat_variance

    def quaternion_batch_norm(self, x):
        r, i, j, k, mu, quat_variance = self.normalise_quaternion(x)
        x = torch.cat((r * self.gamma, i * self.gamma, j * self.gamma, k * self.gamma), dim=1) + self.beta
        return x, mu, quat_variance

    def forward(self, x):
        # print(self.training)
        if self.training:

            x, mu, quat_variance = self.quaternion_batch_norm(x)

            # with torch.no_grad():
            self.moving_mean.copy_(moving_average_update(self.moving_mean.data, mu, self.momentum))
            self.moving_var.copy_(moving_average_update(self.moving_var.data, quat_variance, self.momentum))

            return x

        else:
            with torch.no_grad():
                # print(input.shape, self.moving_mean.shape)
                r, i, j, k = torch.chunk(x, 4, dim=1)
                quaternions = [r, i, j, k]
                output = []
                denominator = torch.sqrt(self.moving_var + self.eps)
                beta_components = torch.chunk(self.beta, 4, dim=1)
                # print(torch.tensor(quaternions).shape)
                # print(quaternions[0].shape, self.moving_mean.shape, self.moving_var.shape, torch.squeeze(self.beta).shape)
                for q in range(4):
                    new_quat = self.gamma * ((quaternions[q] - self.moving_mean[q]) / denominator) + beta_components[q]
                    output.append(new_quat)
                output = torch.cat(output, dim=1)

                return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
