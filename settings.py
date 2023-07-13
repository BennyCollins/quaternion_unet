import os


def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Variables relating to data processing
ORIGINAL_SAMPLING_RATE = 44100
TARGET_SAMPLING_RATE = 22050         # Set the downsampling rate.
SAMPLING_RATE_STR = '_' + str(TARGET_SAMPLING_RATE)
WAV_WRITE_BITS = 'PCM_16'            # Set the number of bits when saving wavs e.g. 'PCM_16', 'PCM_24'
DURATION = 6  # in seconds           # Set the duration of sample excerpt
NFFT = 1024                          # Set the NFFT parameter for STFT
OVERLAP_FACTOR = 4
HOP_LENGTH = NFFT // OVERLAP_FACTOR  # Set the Hop Length parameter for STFT
OVERLAP = NFFT - HOP_LENGTH          # Number of overlapping frames for consecutive windows in STFT
#FRAMES_PER_SAMPLE = TARGET_SAMPLING_RATE * DURATION
FRAMES_PER_SAMPLE = 65536            # Frames per sample chunk
MONO = False                         # Mono = True, Complex as Channel = False
ENERGY_THRESHOLD = 0                 # Set the energy threshold for considering a sample as silent when filtering.


# File path variables
MAIN_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

MUSDB_FOLDER_PATH = os.path.join(MAIN_DIR_PATH, 'dataset', 'musdb')
MUSDB_WAVS_FOLDER_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb18hq')
MUSDB_CHUNKS_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb_chunks' + SAMPLING_RATE_STR)
ENERGY_PROFILE_PATH = os.path.join(MUSDB_FOLDER_PATH, 'musdb_chunks_energy' + SAMPLING_RATE_STR)
FILTERED_SAMPLE_PATHS = os.path.join(MUSDB_FOLDER_PATH, 'filtered_paths' + SAMPLING_RATE_STR)

INPUT_WAVS_FOLDER = os.path.join(MAIN_DIR_PATH, 'apply', 'input_wavs')

RESULTS_FOLDER = os.path.join(MAIN_DIR_PATH, 'model_results')


SOURCES = ['vocals', 'accompaniment', 'drums', 'bass', 'other']


# Variables for specifying desired sources for model to isolate
TYPE = 'vocals'                      # Set to either '2src' or '4src' or one of the elements of SOURCES

if TYPE == '2src':
    SOURCES_SUBSET = ['vocals', 'accompaniment']
elif TYPE == '4src':
    SOURCES_SUBSET = ['vocals', 'drums', 'bass', 'other']
elif TYPE in SOURCES:
    SOURCES_SUBSET = [TYPE]
else:
    print('TYPE must be one of \'2src\', \'4src\', \'vocals\', \'accompaniment\', \'drums\', \'bass\' or \'other\'.')
SOURCES_SUBSET_ID = [SOURCES.index(i) for i in SOURCES_SUBSET]


# Model building - options and hyperparameters
FILTER_TRAIN_SET = False
FILTER_VAL_SET = False
TRAIN_STANDARD_TEST_MODEL = False    # Train simplified standard unet model
TRAIN_QUATERNION_TEST_MODEL = True  # Train simplified quaternion unet model
TRAIN_STANDARD_MODEL = False          # Train standard unet model
TRAIN_QUATERNION_MODEL = False        # Train quaternion unet model
NUM_CHANNELS_STANDARD = 64           # Num channels after first layer of standard unet
NUM_LAYERS_STANDARD = 5              # Num layers in standard unet
NUM_CHANNELS_QUATERNION = 64         # Num channels after first layer of quaternion unet
NUM_LAYERS_QUATERNION = 5            # Num layers in quaternion unet
SPEC_OUTPUT = False                  # Train models to produce spectrogram output


# Training - options and hyperparameters
USE_CUDA = False                     # Use cuda processing if available
BATCH_SIZE = 16                      # Set the batch size
EPOCHS = 2                           # Set the maximum number of epochs
LR = 0.001                           # Set the learning rate
BETAS = None
WEIGHT_DECAY = None
REPORT_PER_N_BATCHES = 20            # Report progress every n batches
APPLY_AFTER_TRAINING = True          # Apply model and save wav output after training model
EVAL_AFTER_TRAINING = True           # Evaluate model after training
EVAL_ON_TEST_SET = True              # Evaluate model on test set
EVAL_ON_TRAIN_SET = True             # Evaluate model on train set
EVAL_ON_VAL_SET = True               # Evaluate model on validation set


# Evaluation options
TEST_CHUNK_DURATION = 1.             # Num secs for window used in calculating evaluation metrics
#BSS_WINDOW_LEN = int(TEST_CHUNK_DURATION * TARGET_SAMPLING_RATE)
BSS_WINDOW_LEN = FRAMES_PER_SAMPLE   # Num frames