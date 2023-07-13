import os

import torch
import librosa
import soundfile as sf
import numpy as np

from utils.utils import create_folder
from dataset.complex_as_channel import spectro, complex_as_channel, ispectro, icomplex_as_channel
from models.TestNets import TestNet, QTestNet
from settings import ORIGINAL_SAMPLING_RATE, TARGET_SAMPLING_RATE, WAV_WRITE_BITS, NFFT, HOP_LENGTH, FRAMES_PER_SAMPLE, \
    MONO, INPUT_WAVS_FOLDER, RESULTS_FOLDER, USE_CUDA


def get_wav(wav_path):
    y = librosa.load(wav_path, sr=ORIGINAL_SAMPLING_RATE, mono=MONO)[0]
    downsampled = librosa.resample(y, orig_sr=ORIGINAL_SAMPLING_RATE,
                                   target_sr=TARGET_SAMPLING_RATE)
    return downsampled


def chunk_wav(wav_file, window_size):
    channels = wav_file.shape[0]
    window = window_size
    M = (np.max(wav_file.shape) // window)
    zero_padding = window - np.max(wav_file.shape) % window
    zeros = np.zeros([channels, zero_padding])
    wav_file = np.concatenate([wav_file, zeros], axis=-1)
    padded_song_length = wav_file.shape[1]
    chunks = np.reshape(wav_file, (channels, M + 1, window))
    chunks = np.swapaxes(chunks, 0, 1)
    return chunks, padded_song_length


def wav_from_chunks(wav_chunks):
    channels = wav_chunks.size(dim=1)
    wav_chunks = torch.swapaxes(wav_chunks, 0, 1)
    wav = torch.reshape(wav_chunks, (channels, -1))
    return wav.T


def apply_model(model, wav_file_name, save_output=True, device='cpu'):
    print(f'Acquiring isolated {model.source} for {wav_file_name} using {model.__class__.__name__}:')

    wav_file_path = os.path.join(INPUT_WAVS_FOLDER, wav_file_name + '.wav')
    wav = get_wav(wav_file_path)
    song_length = wav.shape[1]
    wav_chunks, padded_song_length = chunk_wav(wav, FRAMES_PER_SAMPLE)

    # Format data to be used as model input
    wav_chunks = spectro(wav_chunks, n_fft=NFFT, hop_length=HOP_LENGTH)
    wav_chunks = complex_as_channel(wav_chunks)
    model.to(device)
    model.train(False)
    with torch.no_grad():
        for i in range(wav_chunks.size(dim=0)):
            chunk = torch.unsqueeze(wav_chunks[i], dim=0)
            chunk = chunk.to(device)
            output = model(chunk)
            if model.spec_output:
                # Convert to audio
                output = icomplex_as_channel(output)
                output = ispectro(output, hop_length=HOP_LENGTH, length=FRAMES_PER_SAMPLE)
            if i == 0:
                full_output = output
            else:
                full_output = torch.cat([full_output, output], dim=0)

    # Construct wav from output chunks
    full_output = wav_from_chunks(full_output)
    # Remove output samples from the timesteps resulting from zero-padding the input
    full_output = full_output[:song_length, :]

    if save_output:
        # Create folder and save output wav
        MODEL_RESULTS_FOLDER_PATH = os.path.join(RESULTS_FOLDER, model.__class__.__name__, model.source,
                                                 model.timestamp)
        output_wav_folder = os.path.join(MODEL_RESULTS_FOLDER_PATH, 'wav_output')
        create_folder(output_wav_folder)
        output_wav_file = os.path.join(output_wav_folder, model.timestamp + wav_file_name + '.wav')
        sf.write(output_wav_file, full_output.cpu(), TARGET_SAMPLING_RATE, WAV_WRITE_BITS)
        print(f'Output saved: {output_wav_file}.')
    return wav, full_output


if __name__ == '__main__':
    if USE_CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print(f'Device: {device}')

    test_net = QTestNet()

    wav_file_name = 'Al James - Schoolboy Facination'
    wav, output = apply_model(test_net, wav_file_name, device=device)