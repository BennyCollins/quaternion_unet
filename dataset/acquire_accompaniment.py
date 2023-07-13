import sys
import os

import librosa
import soundfile as sf
import numpy as np

from settings import ORIGINAL_SAMPLING_RATE, MONO, MUSDB_WAVS_FOLDER_PATH

sys.path.append('../')


def song_accompaniment_wavs(folder_path, samplerate):
    # Load wav components of accompaniment
    bass, _ = librosa.load(os.path.join(folder_path, 'bass.wav'), sr=samplerate, mono=MONO)
    drums, _ = librosa.load(os.path.join(folder_path, 'drums.wav'), sr=samplerate, mono=MONO)
    other, _ = librosa.load(os.path.join(folder_path, 'other.wav'), sr=samplerate, mono=MONO)
    # Add components to acquire accompaniment array
    accompaniment = bass + drums + other
    if not MONO:
        # If the audio is stereo, format the array so that it can be saved as a wav
        accompaniment = np.transpose(accompaniment)
    # Save as wav
    sf.write(os.path.join(folder_path, 'accompaniment.wav'), accompaniment, samplerate, 'PCM_16')


def dataset_accompaniment_wavs(dataset_path, samplerate):
    # Iterate through subsets
    for subset in ['test', 'train']:
        # Acquire subset wav folder path
        wav_subset_path = os.path.join(dataset_path, subset)
        # List names of track folders
        track_names = os.listdir(wav_subset_path)
        for track_name in track_names:
            # Acquire track folder path
            track_path = os.path.join(wav_subset_path, track_name)
            # Obtain and save accompaniment.wav for track
            song_accompaniment_wavs(track_path, samplerate)


if __name__ == '__main__':
    dataset_accompaniment_wavs(MUSDB_WAVS_FOLDER_PATH, ORIGINAL_SAMPLING_RATE)