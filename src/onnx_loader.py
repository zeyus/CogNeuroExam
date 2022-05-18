import logging
from time import sleep
from online.util import OnnxOnline, DN3D1010
from eeg.eeg import EEG, Filtering
from psypy import config
import numpy as np



logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

target_sr = 64
sliding_win_size_seconds = 0.5
sliding_step = int(sliding_win_size_seconds / 2)

# order matters

# ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'emg', 'eeg', 'eeg', 'stim']
ch_types = config.data.BCI_CHANNEL_TYPES
# ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'EMG1', 'P3', 'P4', 'marker']
ch_names = config.data.BCI_CHANNEL_NAMES
ch_idx = config.data.BCI_CHANNEL_INDEXES
use_ch = ['P3', 'P4', 'C3', 'C4']
use_ch_idx = [x for x in range(len(ch_names)) if ch_names[x] in use_ch]
# fake for now
# eeg_channels_inds = [0, 1, 2, 3]


# Load the ONNX model
# model = onnx.load("trained_models/2022-05-08_21-38-01_EEGNetStridedOnnxCompat_a.onnx")
model_path = "trained_models/2022-05-18_19-17-46_EEGNetStridedOnnxCompat_a.onnx"

# Check that the model is well formed
# onnx.checker.check_model(model)
model = OnnxOnline(model_path)

chmap = DN3D1010(ch_names, ch_types, use_ch_idx, -0.3, 0.3)




# for now let's simulate data

eeg_source = EEG(dummyBoard = True)
eeg_sr = eeg_source.sampling_rate
eeg_filter = Filtering(use_ch_idx, eeg_sr)
eeg_source.start_stream()
sleep(sliding_win_size_seconds)

# need to change this to a buffer system
data = np.zeros((len(ch_idx), 0))
while True:
  # already throw away irrelevant channels
  data_chunk = eeg_source.poll()[ch_idx, :]
  data = np.concatenate((data, chmap.zero_discard(data_chunk)), axis=1)
  if data.shape[1] > eeg_sr * sliding_win_size_seconds:
    # bandpass, then downsample
    data = eeg_filter.bandpass(data, 8, 32)
    print(data.shape)
    data = eeg_filter.resample(data.T, target_sr)

    # do NOT keep this hardcoded
    # in fact, drop  useless channels earlier
    # even turn them off on board if possible.
    data_predict = chmap(data)
    print(data_predict.shape)
    print(model.predict())

  sleep(sliding_step)
