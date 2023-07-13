import os
import shutil
from pathlib import PureWindowsPath

from sklearn.model_selection import train_test_split

from utils.utils import create_folder
from settings import MUSDB_CHUNKS_PATH


'''
The latter part of the __main__ method (from the call to train_test_split and onwards) is taken from the url below:
repo url: https://github.com/vskadandale/multichannel-unet-bss
file url: https://github.com/vskadandale/multichannel-unet-bss/blob/master/dataset/preprocessing.py

@inproceedings{kadandale2020multi,
  title={Multi-channel U-Net for Music Source Separation},
  author={Kadandale, Venkatesh S and Montesinos, Juan F and Haro, Gloria and G{\'o}mez, Emilia},
  booktitle={2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
'''


def list_absolute_paths(root):
    abs_paths = []
    # Iterate through names of files and directories
    for name in os.listdir(root):
        # Append full path to list
        abs_paths.append(os.path.abspath(os.path.join(root, name)))
    return abs_paths


if __name__ == '__main__':
    # Get train set path and validation destination folder path
    TRAIN_PATH = os.path.join(MUSDB_CHUNKS_PATH, 'train')
    VALIDATION_PATH = os.path.join(MUSDB_CHUNKS_PATH, 'val')
    create_folder(VALIDATION_PATH)

    # Acquiring training audio directory paths
    train_track_dirs = list_absolute_paths(TRAIN_PATH)
    train_file_paths = []
    # Iterate through directory paths and append absolute audio file paths to list
    for directory in train_track_dirs:
        train_file_paths += list_absolute_paths(directory)

    # Creating validation set
    _, val_paths = train_test_split(train_file_paths, test_size=0.05, random_state=0)
    # Iterate through file paths in validation split
    for file in val_paths:
        file = PureWindowsPath(file)
        # Replace 'train' subset folder name with 'val' to obtain destination folder
        val_path = str.replace(file.as_posix(), 'train', 'val')
        # Create validation track directory
        create_folder(os.path.abspath(os.path.join(val_path, os.pardir)))
        # Move audio chunk from train folder to validation folder
        shutil.move(file, val_path)
