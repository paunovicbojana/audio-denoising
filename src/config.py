"""
config.py
---------
Central configuration file for the audio denoising project.
All hyperparameters and paths are defined here so you only
need to change one file when experimenting.
"""

import os

#----------------------------------------------------------------------------------
# Paths
#----------------------------------------------------------------------------------

# Root of the downloaded DNS dataset
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

# Paths to raw WAV files
CLEAN_DIR  = os.path.join(DATA_ROOT, "raw/clean_fullband/mnt/dnsv5/clean")    # clean speech WAVs
NOISE_DIR  = os.path.join(DATA_ROOT, "raw/noise_fullband")                    # background noise WAVs

# Dev testset (noisy/clean pairs from DNS)
DEV_NOISY_DIR = os.path.join(DATA_ROOT, "processed", "noisy_testclips")
DEV_CLEAN_DIR = os.path.join(DATA_ROOT, "processed", "clean_testclips")

# Where preprocessed .pt segment files will be saved
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Where trained model checkpoints are saved
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# Where denoised output WAVs are written during inference
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


#----------------------------------------------------------------------------------
# Audio settings
#----------------------------------------------------------------------------------

SAMPLE_RATE    = 16000   # 16kHz, standard for DNS dataset
N_FFT          = 512     # FFT window size -> frequency resolution
HOP_LENGTH     = 128     # hop between consecutive STFT frames (25% of N_FFT)
WIN_LENGTH     = 512     # analysis window length (same as N_FFT)

# After STFT the frequency axis has N_FFT//2 + 1 = 257 bins
N_FREQ_BINS = N_FFT // 2 + 1


#----------------------------------------------------------------------------------
# Segmentation
#----------------------------------------------------------------------------------

SEGMENT_DURATION = 2.0                                   # seconds per training segment
SEGMENT_SAMPLES  = int(SAMPLE_RATE * SEGMENT_DURATION)   # 32000 samples
OVERLAP          = 0.5                                   # 50% overlap -> new segment every 1 second


#----------------------------------------------------------------------------------
# Dataset split
#----------------------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Splits are done by speaker_id so no speaker appears in more than one split

# Max segments to load per split
# Set to None to use the full dataset
MAX_TRAIN_FILES = None
MAX_VAL_FILES   = None


#----------------------------------------------------------------------------------
# Model architecture
#----------------------------------------------------------------------------------

ENCODER_CHANNELS = [32, 64, 128, 256, 512]    # doubled vs original - more capacity
LSTM_HIDDEN      = 512                        # hidden units per direction in BiLSTM
LSTM_LAYERS      = 2                          # number of stacked BiLSTM layers


#----------------------------------------------------------------------------------
# Training
#----------------------------------------------------------------------------------

BATCH_SIZE    = 32      # larger batch = more stable gradients, faster per epoch
MAX_EPOCHS    = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4

# Warmup + cosine annealing schedule
WARMUP_EPOCHS   = 3     # faster warmup so model reaches full LR sooner
MIN_LR          = 1e-5  # don't decay LR too far - keeps learning at later epochs

# Early stopping: stop if validation SI-SDR does not improve for this many epochs
EARLY_STOP_PATIENCE = 15

# Data augmentation: randomly shift SNR by ±SNR_AUGMENT_DB each batch
SNR_AUGMENT_DB = 3.0


#----------------------------------------------------------------------------------
# Loss function weights
#----------------------------------------------------------------------------------

STFT_LOSS_WEIGHT = 0.1   # L = -SI_SDR + 0.1 * STFT_loss

# Multi-resolution STFT loss: list of (n_fft, hop_length, win_length) tuples
STFT_RESOLUTIONS = [
    (256,  64,  256),
    (512,  128, 512),
    (1024, 256, 1024),
]


#----------------------------------------------------------------------------------
# Weights & Biases experiment tracking (set to None to disable)
#----------------------------------------------------------------------------------

WANDB_PROJECT = "audio-denoising"
WANDB_ENTITY  = None                # your W&B username, or None to use default