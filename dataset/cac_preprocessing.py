import sys
import os
from functools import partial

import librosa
import librosa.display
import numpy as np
import pandas as pd

from utils.utils import create_folder
from settings import MUSDB_WAVS_FOLDER_PATH, MUSDB_CHUNKS_PATH, ENERGY_PROFILE_PATH, ORIGINAL_SAMPLING_RATE, \
    TARGET_SAMPLING_RATE, FRAMES_PER_SAMPLE, MONO, SOURCES

sys.path.append('../../')

'''
get_sources, split_sources, get_signal_energy functions are all taken from the url below:
repo url: https://github.com/vskadandale/multichannel-unet-bss
file url: https://github.com/vskadandale/multichannel-unet-bss/blob/master/dataset/preprocessing.py

while get_chunk_energy and the __main__ method are both heavily adapted from material in the same file.

@inproceedings{kadandale2020multi,
  title={Multi-channel U-Net for Music Source Separation},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria and G{\'o}mez, Emilia},
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
'''


def get_sources(track_folder):
    # Create list by iterating through each source, and the full mix, and loading the wav for that source
    y = [librosa.load(os.path.join(track_folder, element + '.wav'),
                      sr=ORIGINAL_SAMPLING_RATE, mono=MONO)[0] for element in [*SOURCES, 'mixture']]
    # Downsample each wav in the list and turn into array
    downsampled = np.stack(list(map(partial(librosa.resample,
                                            orig_sr=ORIGINAL_SAMPLING_RATE,
                                            target_sr=TARGET_SAMPLING_RATE), y)))
    return downsampled


def split_sources(sources, flag):
    # Acquire number of channels
    channels = sources.shape[1]
    # Acquire number of chunks in track before padding, for a number of frames specified in settings.py
    M = (np.max(sources.shape) // FRAMES_PER_SAMPLE)

    if flag == 'train':
        # Take maximum number of chunks before padding
        splits = sources[..., :M * FRAMES_PER_SAMPLE]
        # Split track into chunks via reshaping
        splits = np.reshape(splits, splits.shape[:-1] + (M, FRAMES_PER_SAMPLE))
    else:
        # Acquire number of zeros for padding and create array of zeros
        zero_padding = FRAMES_PER_SAMPLE - np.max(sources.shape) % FRAMES_PER_SAMPLE
        zeros = np.zeros([len(SOURCES) + 1, channels, zero_padding])
        # Pad audio by concatenating audio with zeros
        splits = np.concatenate([sources, zeros], axis=2)
        # Split track into chunks via reshaping
        splits = np.reshape(splits, splits.shape[:-1] + (M + 1, FRAMES_PER_SAMPLE))
    return splits


def get_signal_energy(signal):
    return sum(abs(signal) ** 2)


def get_chunk_energy(chunk_id, track_name, sources, energy_df=None):
    # Create dictionary for storing audio source energy info
    energy_dict = {}
    # Iterate through sources
    for source_id, source in enumerate([*SOURCES, 'MIX']):
        # Acquire signal
        signal = sources[source_id, :, chunk_id, :]
        if not MONO:
            # Convert signal to mono
            signal = np.mean(signal, 0)
        # Add source energy to dictionary
        energy_dict[source] = str(int(round(get_signal_energy(signal))))
    # Create dataframe index name for this chunk
    index_str = track_name + ' (' + str(chunk_id) + ')'
    # Create a dataframe containing the energy data for the specified chunk
    chunk_energy_df = pd.DataFrame(energy_dict, index=[index_str])
    if energy_df is None:
        # If the dataframe doesn't exist, return the chunk dataframe
        return chunk_energy_df
    else:
        # Otherwise, return a concatenation of the subset and chunk dataframes
        return pd.concat([energy_df, chunk_energy_df])


if __name__ == '__main__':

    for subset_type in ['train', 'test']:
        # Acquire subset wav folder path and sorted list of track names
        subset_wav_path = os.path.join(MUSDB_WAVS_FOLDER_PATH, subset_type)
        tracks = sorted(os.listdir(subset_wav_path))
        # Define new energy dataframe variable for subset
        subset_energy_df = None
        for track_id, track_name in enumerate(tracks):
            # Acquire track wavs folder path
            track_path = os.path.join(subset_wav_path, track_name)
            # Acquire path of destination folder for saving audio arrays
            dump_path = os.path.join(MUSDB_CHUNKS_PATH, subset_type, track_name)
            create_folder(dump_path)
            print(f'Loading and downsampling track [{track_id + 1}/{len(tracks)}] || {track_name}')
            # Downsample audio for track audio sources
            sources_downsampled = get_sources(track_path)
            # Split the track's audio sources into chunks of length specified in settings.py
            track_chunks = split_sources(sources_downsampled, subset_type)
            # Iterate through chunks
            for chunk_id in range(track_chunks.shape[2]):
                print(
                    f'Subset [{subset_type}] || Chunk [{chunk_id + 1}/{track_chunks.shape[2]}] || Track [{track_id + 1}/{len(tracks)}]')
                # Acquire matrix of sources for chunk
                array = track_chunks[..., chunk_id, :]
                # Acquire file destination path and save chunk array
                full_path = os.path.join(dump_path, str(chunk_id))
                np.save(full_path, array)
                # Update subset energy dataframe
                subset_energy_df = get_chunk_energy(chunk_id, track_name, track_chunks, energy_df=subset_energy_df)
            del track_chunks
        # Create filepath to save subset energy dataframe and save as csv
        energy_file_path = os.path.join(ENERGY_PROFILE_PATH, subset_type + '.csv')
        subset_energy_df.to_csv(energy_file_path)
