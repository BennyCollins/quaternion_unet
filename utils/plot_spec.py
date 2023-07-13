import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.TestNets import TestNet
from dataset.cac_dataloader import CACUnetInput
from utils.utils import create_folder
from settings import NFFT, TARGET_SAMPLING_RATE, OVERLAP, NUM_CHANNELS_QUATERNION, NUM_LAYERS_QUATERNION, TYPE, SPEC_OUTPUT


def plot_chunk_specs(model, model_directory, device='cpu'):
    # Create directory in which to save spectrograms
    spec_directory = os.path.join(model_directory, 'spectrograms')
    create_folder(spec_directory)

    # Create data set that we will take 1 sample from
    data_set = CACUnetInput('train', model.source, filter_train_val=True)
    sample_input, sample_target = data_set[0]
    # Add an extra dimension to acquire model input format
    sample_input = torch.unsqueeze(sample_input, 0)
    sample_input = sample_input.to(device)
    model = model.to(device)
    # Acquire output
    with torch.no_grad():
        sample_output = model(sample_input)
    # Format output for plotting
    sample_output = torch.squeeze(sample_output).cpu().numpy()

    # Convert audio to mono by taking average over audio channels
    sample_output = np.mean(sample_output, axis=0)
    sample_target = np.mean(sample_target, axis=0)

    # Plot and save magnitude spectrograms
    fig, ax = plt.subplots(2)
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Sample chunk magnitude spectrograms for \nmodel trained to isolate {model.source}', fontsize=12)
    ax[0].specgram(sample_output, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP, mode='magnitude')
    ax[0].set_title('Model output', fontsize=10)
    ax[0].set_xlabel('Time', fontsize=8)
    ax[0].set_ylabel('Frequency', fontsize=8)
    ax[1].specgram(sample_target, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP, mode='magnitude')
    ax[1].set_title('Target spectrogram', fontsize=10)
    ax[1].set_xlabel('Time', fontsize=8)
    ax[1].set_ylabel('Frequency', fontsize=8)
    plt.savefig(os.path.join(spec_directory, model.timestamp + 'mag_specs.jpg'))

    # Plot and save phase spectrograms
    fig, ax = plt.subplots(2)
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Sample chunk phase spectrograms for \nmodel trained to isolate {model.source}', fontsize=12)
    ax[0].specgram(sample_output, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP, mode='phase')
    ax[0].set_title('Model output', fontsize=10)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Frequency')
    ax[1].specgram(sample_target, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP, mode='phase')
    ax[1].set_title('Target spectrogram', fontsize=10)
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Frequency')
    plt.savefig(os.path.join(spec_directory, model.timestamp + 'phase_specs.jpg'))

    # Plot and save power spectral density spectrograms
    fig, ax = plt.subplots(2)
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Sample chunk power spectral density spectrograms \nfor model trained to isolate {model.source}',
                 fontsize=12)
    ax[0].specgram(sample_output, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP)
    ax[0].set_title('Model output', fontsize=10)
    ax[0].set_xlabel('Time', fontsize=8)
    ax[0].set_ylabel('Frequency', fontsize=8)
    ax[1].specgram(sample_target, Fs=TARGET_SAMPLING_RATE, NFFT=NFFT, noverlap=OVERLAP)
    ax[1].set_title('Target spectrogram', fontsize=10)
    ax[1].set_xlabel('Time', fontsize=8)
    ax[1].set_ylabel('Frequency', fontsize=8)
    plt.savefig(os.path.join(spec_directory, model.timestamp + 'pse_specs.jpg'))

    print(f'Spectrograms saved to {spec_directory}')


if __name__ == '__main__':
    model = TestNet(num_channels=NUM_CHANNELS_QUATERNION, num_layers=NUM_LAYERS_QUATERNION, source=TYPE,
                    spec_output=SPEC_OUTPUT)
    plot_chunk_specs(model)