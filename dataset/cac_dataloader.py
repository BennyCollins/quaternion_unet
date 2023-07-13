import random
import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from dataset.complex_as_channel import spectro, complex_as_channel
from settings import NFFT, HOP_LENGTH, MUSDB_CHUNKS_PATH, FILTERED_SAMPLE_PATHS, SOURCES


class CACUnetInput(torch.utils.data.Dataset):
    '''Dataset class for single-source separating models'''

    def __init__(self, state, source, spec_output=False, filter_train_val=False):
        '''
        state = desired dataset ('train', 'val' or 'test')
        source = desired source to separate ('vocals', 'accompaniment', 'drums', 'bass' or 'other')
        spec_output = desired model output (True = spectrogram, False = audio)
        filter_train_val = filter out silent target audio from training or validation set (test set is always filtered)
        '''
        self.spec_output = spec_output
        self.filter_train_val = filter_train_val

        # Acquire for indices of sources to be removed from audio array
        try:
            self.remove_source_ids = np.setdiff1d(np.arange(len(SOURCES)), SOURCES.index(source))
        except ValueError as e:
            raise Exception(
                'source parameter must be one of \'vocals\', \'accompaniment\', \'drums\', \'bass\' or \'other\'.')

        # Define list of files to be loaded by dataloader
        self.input_list = []
        # List all the .npy files in the directory
        paths = list(Path(os.path.join(MUSDB_CHUNKS_PATH, state)).rglob("*.npy"))
        # If class is for unfiltered training or validation sets
        if state != 'test' and not filter_train_val:
            # Iterate through file paths and append formatted path to input list
            for filepath in paths:
                filepath_str = filepath.as_posix()
                self.input_list.append(filepath_str)
        else:
            # Load list of filtered paths
            filtered_paths = np.load(os.path.join(FILTERED_SAMPLE_PATHS, state, source + '_filtered') + '.npy')
            # Iterate through file paths
            for filepath in paths:
                filepath_str = filepath.as_posix()
                # If the chunk path is in list of filtered chunks, append formatted path to input list
                if filepath_str in filtered_paths:
                    self.input_list.append(filepath_str)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        # Load audio array from chunk path in input list
        samples = np.load(self.input_list[idx], allow_pickle=True)
        # Filter out undesired source stems
        filtered_samples = np.delete(samples, self.remove_source_ids, axis=0)
        # If the model has a spectrogram output
        if self.spec_output:
            # Take STFT
            stft_output = spectro(filtered_samples, n_fft=NFFT, hop_length=HOP_LENGTH)
            # Convert STFT output to complex-as-channel form
            stft_output = complex_as_channel(stft_output)
            # Isolate mixture spectrogram
            mixture_spec = stft_output[-1]
            # Isolate target spectrogram
            target_spec = stft_output[0]
            return mixture_spec, target_spec
        else:
            # Isolate target audio
            target_audio = filtered_samples[0]
            # Take STFT of mixture audio
            mixture_spec = spectro(filtered_samples[-1], n_fft=NFFT, hop_length=HOP_LENGTH)
            # Convert STFT output to complex-as-channel form
            mixture_spec = complex_as_channel(mixture_spec)
            return mixture_spec, target_audio


class CACMultiSourceUnetInput(torch.utils.data.Dataset):
    def __init__(self, state, type, spec_output=False, filter_train_val=False):
        '''
        state = desired dataset ('train', 'val' or 'test')
        type = '2src' or '4src' (singing voice separation or multi-instrument separation)
        spec_output = desired model output (True = spectrogram, False = audio)
        filter_train_val = filter out silent target audio from training or validation set (test set is always filtered)
        '''
        self.spec_output = spec_output

        # Acquire subset of desired isolated sources
        if type == '2src':
            sources_subset = ['vocals', 'accompaniment']
        elif type == '4src':
            sources_subset = ['vocals', 'drums', 'bass', 'other']
        # Acquire ID number of each source
        sources_subset_id = [SOURCES.index(i) for i in sources_subset]

        # Acquire for indices of sources to be removed from audio array
        try:
            self.remove_source_ids = np.setdiff1d(np.arange(len(SOURCES)), sources_subset_id)
        except ValueError as e:
            raise Exception('type parameter must be one of \'2src\', or \'4src\'.')

        # Define list of files to be loaded by dataloader
        self.input_list = []
        # List all the .npy files in the directory
        paths = list(Path(os.path.join(MUSDB_CHUNKS_PATH, state)).rglob("*.npy"))
        # If class is for unfiltered training or validation sets
        if state != 'test' and not filter_train_val:
            # Iterate through file paths and append formatted path to input list
            for filepath in paths:
                filepath_str = filepath.as_posix()
                self.input_list.append(filepath_str)
        else:
            # Load list of filtered paths
            filtered_paths = np.load(os.path.join(FILTERED_SAMPLE_PATHS, state, type + '_filtered') + '.npy')
            # Iterate through file paths
            for filepath in paths:
                filepath_str = filepath.as_posix()
                # If the chunk path is in list of filtered chunks, append formatted path to input list
                if filepath_str in filtered_paths:
                    self.input_list.append(filepath_str)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = np.load(self.input_list[idx], allow_pickle=True)
        filtered_samples = np.delete(sample, self.remove_source_ids, axis=0)
        # If the model has a spectrogram output
        if self.spec_output:
            # Take STFT
            stft_output = spectro(filtered_samples, n_fft=NFFT, hop_length=HOP_LENGTH)
            # Convert STFT output to complex-as-channel form
            stft_output = complex_as_channel(stft_output)
            # Isolate mixture spectrogram
            mixture_spec = stft_output[-1]
            # Isolate array of target source spectrograms
            target_specs = stft_output[:-1]
            return mixture_spec, target_specs
        else:
            # Isolate array of target sources' audio
            target_audios = filtered_samples[:-1]
            # Take STFT of mixture audio
            mixture_spec = spectro(filtered_samples[-1], n_fft=NFFT, hop_length=HOP_LENGTH)
            # Convert STFT output to complex-as-channel form
            mixture_spec = complex_as_channel(mixture_spec)
            return mixture_spec, target_audios


if __name__ == '__main__':

    for subset in ['train', 'test', 'val']:
        num_chunks = {}
        for source in ['vocals', 'drums', 'bass', 'other', 'accompaniment']:
            data_set = CACUnetInput(subset, source, filter_train_val=True)
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)
            num_chunks[source] = len(data_loader)
            data_set[0]

        for type in ['2src', '4src']:
            data_set = CACMultiSourceUnetInput(subset, type, filter_train_val=True)
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)
            num_chunks[type] = len(data_loader)

        data_set = CACUnetInput(subset, 'vocals')
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0)
        num_chunks['unfiltered'] = len(data_loader)

        if subset == 'train':
            num_chunks_df = pd.DataFrame(num_chunks, index=[subset])
        else:
            subset_num_chunks_df = pd.DataFrame(num_chunks, index=[subset])
            num_chunks_df = num_chunks_df.append(subset_num_chunks_df)

    print(f'Number of chunks in subsets with and without filtering: \n{num_chunks_df}')
    #num_chunks_df.to_csv(os.path.join(MUSDB_FOLDER_PATH, f'num_chunks_{TARGET_SAMPLING_RATE}.csv'))
