
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.utils import create_folder
from settings import MUSDB_CHUNKS_PATH, ENERGY_PROFILE_PATH, FILTERED_SAMPLE_PATHS, SOURCES

sys.path.append('..')


'''
filter_dataset is heavily adapted from material in the file url below:
repo url: https://github.com/vskadandale/multichannel-unet-bss
file url: https://github.com/vskadandale/multichannel-unet-bss/blob/master/dataset/filter_musdb_split.py

@inproceedings{kadandale2020multi,
  title={Multi-channel U-Net for Music Source Separation},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria and G{\'o}mez, Emilia},
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
'''


'''
This version of filter_musdb_split is adapted for execution on the remote hyperion cluster
'''


def filter_dataset(subset, dataset_type):
    # Create a list to store filenames for audio chunks with no silent desired sources
    selected_files = []

    # Acquire subset of audio source names for dataset type
    if dataset_type == '2src':
        sources_subset = ['vocals', 'accompaniment']
    elif dataset_type == '4src':
        sources_subset = ['vocals', 'drums', 'bass', 'other']
    elif dataset_type in SOURCES:
        sources_subset = [dataset_type]

    # Acquire subset energy profile file path and load into dataframe
    if subset == 'test':
        energy_file_path = os.path.join(ENERGY_PROFILE_PATH, 'test.csv')
    else:
        # Validation set energy values are also in train.csv
        energy_file_path = os.path.join(ENERGY_PROFILE_PATH, 'train.csv')
    subset_energy_df = pd.read_csv(energy_file_path, index_col=0)
    # Filter dataframe columns for relevant sources
    subset_energy_df = subset_energy_df[sources_subset]

    # Acquire path for audio chunks subset
    subset_path = os.path.join(MUSDB_CHUNKS_PATH, subset)
    # Acquire list of track names
    tracks = os.listdir(subset_path)
    # Iterate through track names
    for track_name in tracks:
        # Acquire track path
        track_path = os.path.join(subset_path, track_name)
        # Acquire list of chunk file names
        track_chunks = os.listdir(track_path)
        # Iterate through chunk names
        for chunk_file_name in track_chunks:
            # Acquire chunk file path
            chunk_path = os.path.join(track_path, chunk_file_name)
            # Obtain chunk index by removing file suffix
            chunk_index_str = chunk_file_name[:-len('.npy')]
            # Acquire index for row containing chunk energy values
            energy_row_index = track_name + ' (' + chunk_index_str + ')'
            # If any of the chunk's energy values are 0
            if not (subset_energy_df.loc[energy_row_index] == 0).any():
                # Format the filepath
                chunk_path = Path(chunk_path)
                # Append file path to the list of filtered chunk paths
                selected_files.append(chunk_path.as_posix())
    # Acquire destination folder path for saving filtered chunk paths
    filtered_subset_folder_path = os.path.join(FILTERED_SAMPLE_PATHS, subset)
    create_folder(filtered_subset_folder_path)
    # Acquire destination file path and save filtered chunk paths
    filtered_subset_file_path = os.path.join(filtered_subset_folder_path, dataset_type + '_filtered')
    np.save(filtered_subset_file_path, selected_files)
    print(f'Saving filtered paths to {filtered_subset_file_path}')
    return


if __name__ == '__main__':
    # Iterate through subsets
    for subset in ['train', 'test', 'val']:
        # Iterate through desired isolated sources
        for dataset_type in ['vocals', 'drums', 'bass', 'other', 'accompaniment', '2src', '4src']:
            # Save csv of dataset filtered chunk paths
            filter_dataset(subset, dataset_type)
