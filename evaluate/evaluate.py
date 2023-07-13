import os
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import museval

from utils.utils import create_folder
from dataset.complex_as_channel import ispectro, icomplex_as_channel
from dataset.cac_dataloader import CACUnetInput
from models.TestNets import TestNet, QTestNet
from models.UNet import UNet
from settings import HOP_LENGTH, RESULTS_FOLDER, BSS_WINDOW_LEN, USE_CUDA

'''
new_sdr and eval_track functions adapted from doe in Hybrid Demucs repo (url below):
repo url: https://github.com/facebookresearch/demucs
file url: https://github.com/facebookresearch/demucs/blob/main/demucs/evaluate.py

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
'''


def get_signal_energy(signal):
    return torch.sum(torch.abs(signal) ** 2)


def new_sdr(references, estimates, win_len):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    n_remainder_samples = np.max(estimates.shape) % win_len
    if n_remainder_samples != 0:
        n_zeros = win_len - n_remainder_samples
        zeros = np.zeros([estimates.size(dim=0), n_zeros, estimates.size(dim=2)])
        references = np.concatenate([references, zeros], axis=1)
        estimates = np.concatenate([estimates, zeros], axis=1)
    scores = []
    for i in range(int(estimates.shape[1] / win_len)):
        references_chunk = references[:, win_len * i: win_len * (i + 1), :]
        estimates_chunk = estimates[:, win_len * i: win_len * (i + 1), :]
        if get_signal_energy(references_chunk) == 0:
            use_chunk = False
        else:
            use_chunk = True
        if use_chunk:
            delta = 1e-7  # avoid numerical errors
            num = torch.sum(torch.square(references_chunk), dim=(-2, -1))
            den = torch.sum(torch.square(references_chunk - estimates_chunk), dim=(-2, -1))
            num += delta
            den += delta
            score = 10 * torch.log10(num / den)
            scores.append(score.item())
    return np.asarray(scores)


def eval_track(references, estimates, win=BSS_WINDOW_LEN, hop=BSS_WINDOW_LEN, only_nsdr=False):
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()
    references = references.cpu()
    estimates = estimates.cpu()

    new_scores = new_sdr(references, estimates, win)

    if only_nsdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def acquire_metric_stats_over_chunks(metric_data, metric_name):
    metric_data = np.squeeze(metric_data)
    if np.isfinite(metric_data).any():
        mean = np.mean(metric_data).item()
        std = np.std(metric_data).item()
        median = np.median(metric_data).item()
        mad = np.median(np.absolute(metric_data - median)).item()
        max = np.amax(metric_data).item()
        min = np.amin(metric_data).item()
    else:
        print(f"{metric_name}: Metric values don't include defined real numbers.")
        mean = np.nan
        std = np.nan
        median = np.nan
        mad = np.nan
        max = np.nan
        min = np.nan
    return {'Mean': mean, 'STD': std, 'Median': median, 'MAD': mad, 'Max': max, 'Min': min}


def acquire_all_metrics_stats_over_chunks(metric_data_dict):
    for i, (metric_name, metric_data) in enumerate(metric_data_dict.items()):
        # Remove NaN values
        metric_data = metric_data[~np.isnan(metric_data)]
        metric_stats = acquire_metric_stats_over_chunks(metric_data, metric_name)
        if i == 0:
            metrics_df = pd.DataFrame(metric_stats, index=[metric_name])
        else:
            metric_df = pd.DataFrame(metric_stats, index=[metric_name])
            metrics_df = metrics_df.append(metric_df)
    print(metrics_df)
    return metrics_df


def eval_model(model, subset='test', only_nsdr=False, device='cpu'):
    eval_set = CACUnetInput(subset, model.source, filter_train_val=True)
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

    num_chunks = len(eval_loader)
    print(f'{subset} set has {num_chunks} chunks of audio.')

    model.train(False)
    with torch.no_grad():
        for i, eval_data in enumerate(eval_loader):
            if (i + 1) % 20 == 0:
                print(
                    f'{model.__class__.__name__} ({model.source}): Acquiring output for chunk [{i + 1}||{num_chunks}]')
            input, target = eval_data
            input, target = input.to(device), target.to(device)
            output = model(input)
            if model.spec_output:
                # Convert to audio
                output = icomplex_as_channel(output)
                output = ispectro(output, hop_length=HOP_LENGTH, length=BSS_WINDOW_LEN)

            # If models are to be evaluated only on the new definition of SDR
            if only_nsdr:
                _, chunk_nsdr = eval_track(target, output, only_nsdr=only_nsdr)
                if i == 0:
                    metric_matrices = {'nSDR': chunk_nsdr}
                else:
                    chunk_metric_matrices = {'nSDR': chunk_nsdr}
                    for metric_name, metric in metric_matrices.items():
                        metric_matrices[metric_name] = np.concatenate([metric, chunk_metric_matrices[metric_name]],
                                                                      axis=0)
            else:
                (chunk_sdr, _, chunk_sir, chunk_sar), chunk_nsdr = eval_track(target, output)
                if i == 0:
                    metric_matrices = {'SDR': chunk_sdr, 'SIR': chunk_sir, 'SAR': chunk_sar,
                                       'nSDR': chunk_nsdr}
                else:
                    chunk_metric_matrices = {'SDR': chunk_sdr, 'SIR': chunk_sir, 'SAR': chunk_sar,
                                             'nSDR': chunk_nsdr}
                    for metric_name, metric in metric_matrices.items():
                        metric_matrices[metric_name] = np.concatenate([metric, chunk_metric_matrices[metric_name]],
                                                                      axis=0)
    print('Acquiring metric stats over all chunks')
    stats_df = acquire_all_metrics_stats_over_chunks(metric_matrices)

    MODEL_RESULTS_FOLDER_PATH = os.path.join(RESULTS_FOLDER, model.__class__.__name__, model.source,
                                             model.timestamp)
    model_results_folder_path = os.path.join(MODEL_RESULTS_FOLDER_PATH, 'test_results')
    create_folder(model_results_folder_path)
    test_results_path = os.path.join(model_results_folder_path, model.timestamp + subset + '_set.csv')
    print(f'Test results saving to {test_results_path}')
    stats_df.to_csv(test_results_path)


if __name__ == '__main__':
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Device: {device}')

    source = 'vocals'
    test_net = UNet(num_channels=64, source='vocals', spec_output=False).to(device)
    file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_model(test_net, subset='train', device=device)
