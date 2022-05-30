import logging
from math import floor
from time import sleep, time_ns
import sys
from datetime import datetime
from online.util import OnnxOnline, DN3D1010
from eeg.eeg import EEG, Filtering, CytonSampleRate
from psypy import config
import numpy as np
import pandas as pd

# need to adjust this to print N meters, 1 per label, with range -1 to 1 that update with each prediction
def printMeters(items, pad = 5, length = 30, fill = '█'):
  n_items = len(items) # 3
  len_per_item = length # floor(length / n_items) # 30 / 3 = 10
  meter_out = '\r\033[s\033[0m'
  reset = '\033[0m'
  print(f'\033[{n_items}A', end = '')
  for value, minv, maxv, prefix, selected, n_recent in items: # (0.5, -1, 1, '', '')
    suffix = '\033[1;96m{}\033[0m' if selected  else '{}'
    suffix = suffix.format(('*' * n_recent).ljust(pad))
    decorate = '\033[1;32m' if maxv > 0 and value > 0.5 else '' if maxv <= 0 else '\033[1;31m'
    p_pos = (value - minv) / (maxv - minv) # (0.5 - -1) / (1 - -1) = 0.75
    pos = max(int(len_per_item * p_pos), 1) # 10 * 0.75 = 7.5 -> 7
    val = ("{:6.2f}").format(value)
    bar = '-' * (pos-1) + decorate + fill + reset + '-' * (len_per_item - pos)
    prefix = prefix.rjust(13)
    meter_out += f'\r\033[1B ‖ {prefix} |{bar}| {decorate}{val}{reset} ‖ {suffix}'
  print(meter_out, end = "\033[u")

logging.basicConfig()
# set to logging.INFO if you want to see less debugging output
logging.getLogger().setLevel(logging.DEBUG)

target_sr = 128
sliding_win_size_seconds = 0.8
window_samples = int(target_sr * sliding_win_size_seconds)
# our striding will depend on perfomance of inference
sliding_step_samples = 1
batch_size = 1
# match what we've seen in training
# this could be optimized
data_scale_min = -0.0927
data_scale_max = 0.1875
filter_buffer_size = 0 # int(target_sr * 0.5)
# data_scale_min = None
# data_scale_max = None
sleep_more = 0
update_meters_every = 5
d1010 = False

# order matters

# ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'emg', 'eeg', 'eeg', 'stim']
ch_types = config.data.BCI_CHANNEL_TYPES
# ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'EMG1', 'P3', 'P4', 'marker']
ch_names = config.data.BCI_CHANNEL_NAMES
ch_idx = config.data.BCI_CHANNEL_INDEXES
# use_ch = ['P3', 'P4', 'C3', 'C4']
# which channels we use to feed the model
use_ch = ['C3', 'C4', 'P3', 'P4']

# index of all EEG channels (not EMG, stim, etc)
eeg_ch_idx = [x for x in range(len(ch_types)) if ch_types[x] == 'eeg']

# get the names of the eeg channels so we can match the use_ch
eeg_ch_names = [ch_names[x] for x in eeg_ch_idx]

# now get the updated index (after slicing, the indexes will change)
use_ch_idx = [x for x in range(len(eeg_ch_names)) if eeg_ch_names[x] in use_ch]



# Load the ONNX model
# model = onnx.load("trained_models/2022-05-08_21-38-01_EEGNetStridedOnnxCompat_a.onnx")
model_path = "trained_models/2022-05-30_12-01-34_EEGNetOnnxCompat_a.onnx"
requires_bp = False
log_results = False
softmax = True

prediction_scale = [0, 1] if softmax else [-50, 50]

# Check that the model is well formed
# onnx.checker.check_model(model)
model = OnnxOnline(model_path)

chmap = DN3D1010(ch_names, ch_types, use_ch_idx, data_scale_min, data_scale_max)

      # 108: "l"
      # 110: "n"
      # 111: "o"
      # 113: "q"
      # 114: "r"
# labels
# out_labels = ["Left", "Neutral", "Right"]
out_labels = ["Neutral", "Right"]
script_run_dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
prediction_file = "output/predictions_{}.csv".format(script_run_dt)
predictions = pd.DataFrame(columns = ['timestamp', 'start_sample'] + out_labels)



eeg_source = EEG(dummyBoard = False)
eeg_source.start_stream(sdcard = False, sr = CytonSampleRate.SR_250)
eeg_sr = eeg_source.sampling_rate
sr_scale_factor = target_sr / eeg_sr
logging.info("Resample scale factor: {}".format(sr_scale_factor))
logging.info("Samples in window: {}, window size in time: {}ms".format(window_samples, sliding_win_size_seconds * 1000))
logging.info("Using channels: {}".format(use_ch))
logging.info("Predicting every {} samples (every {}ms)".format(sliding_step_samples, sliding_step_samples / target_sr * 1000))
logging.info("Sample rates: - Source: {}, - Target: {}".format(eeg_sr, target_sr))
eeg_filter = Filtering(use_ch_idx, eeg_sr)
sleep(max(sliding_win_size_seconds, filter_buffer_size))
last_pred_label = None
# need to change this to a buffer system
data = np.zeros((len(ch_idx), 0))
# add time tracking
# ...
sample_start_offset = 0
# this window code is garbage, it's just approximating and deleting from the original data
# better to base slices on original sample rate i think.
recent_preds = [
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
]
preds_since_print = 0
while True:
  try:
    # ch_idx will get ALL channels
    data_chunk = eeg_source.poll()[ch_idx, :]
    data = np.concatenate((data, data_chunk), axis=1) # concat may be memory inefficient 
    # if we have enough to start processing
    if floor(sr_scale_factor * data.shape[1]) > window_samples:
      # bandpass, then downsample
      # might consider bessel or chebychev
      if requires_bp:
        data_predict = eeg_filter.bandpass(data, 8, 32)
        data_predict = eeg_filter.resample(data_predict.T, target_sr).T # this has GOT to be inefficient
      else:
        data_predict = eeg_filter.resample(data.T, target_sr).T
      # now subtract average signal of EEG channels from each channel
      # this will also subtract from stim and EMG channels but who cares
      # we're dropping that data anyway
      # data_predict = data_predict - np.mean(data_predict[eeg_ch_idx, :], axis=0)

      # now let's only get the channels we want
      if d1010:
        data_predict = chmap.zero_discard(data_predict)
      else:
        data_predict = data_predict[use_ch_idx, :]

      max_strides = floor((data_predict.shape[1] - window_samples) / sliding_step_samples)
      leftover_samples = (data_predict.shape[1] - window_samples) % sliding_step_samples
      # if leftover_samples > 0:
      #   max_strides += 1
      # logging.debug('Stride/window start time: {}, strides: {}'.format(sample_start_offset / target_sr, max_strides))
      predicted_label = None
      for stride in range(max_strides + 1):
        stride_by = sliding_step_samples
        # get the data for this stride
        start_idx = stride * stride_by
        # do NOT keep this hardcoded
        # in fact, drop  useless channels earlier (we can't if we want average though...)
        # even turn them off on board if possible.
        data_stride = data_predict[:, start_idx:start_idx+window_samples]
        if d1010:
          data_stride = chmap(data_stride)
        # fake batch for now, would be better to just batch all strides!
        data_stride = np.expand_dims(data_stride, axis=0)
        preds = model.predict(data_stride)
        #print(preds)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        # print(preds)
        # average over last dimensions
        while len(preds.shape) >= 3:
            preds = preds.mean(axis=-1)
        #print(preds)
        # if softmax:
        #   preds = np.exp(preds)/np.sum(np.exp(preds), axis=-1)
        # print(preds)
        # print(np.exp(preds))
        # print(np.exp(preds)/np.sum(np.exp(preds), axis=-1))
        # print(np.log(np.exp(preds)/np.sum(np.exp(preds), axis=-1)))
        # eeg_source.stop()
        # exit()
        if log_results:
          predictions.loc[len(predictions)] = [time_ns(), sample_start_offset + start_idx, preds[0][0], preds[0][1], preds[0][2]]
        #preds = np.average(preds[0][0], axis=1)
        
        #preds[0][0] += 10
        #preds[0][1] += 3
        # preds[0][2] += -12.1
        # preds[0] += 3.00
        # preds[1] += 0.6
        # preds[2] -= 7.2
        # print(preds.shape)
        # print(preds)
        # no batch atm
        # 
        pr_idx = int(preds.argmax(axis=-1).mean().item())
        # print(preds)
        # print(pr_idx)
        # eeg_source.stop()
        # exit()
        recent_preds.pop(0)
        recent_preds.append(pr_idx)
        predicted_label = out_labels[pr_idx]
        preds_since_print += 1
        if not last_pred_label == predicted_label:
          # logging.info("Prediction: {}".format(predicted_label))
          # logging.info(f'r:{preds[0]}, n:{preds[1]}, l:{preds[2]}')
          last_pred_label = predicted_label
        if preds_since_print >= update_meters_every:
          if not log_results:
            printMeters([
              (preds[0][0], prediction_scale[0], prediction_scale[1], out_labels[0], out_labels[0] == predicted_label, recent_preds.count(0)),
              #(preds[3], 501050 10, out_labels[3], out_labels[3] == predicted_label, recent_preds.count(3)),
              (preds[0][1], prediction_scale[0], prediction_scale[1], out_labels[1], out_labels[1] == predicted_label, recent_preds.count(1)),
              #(preds[4], 501050 10, out_labels[4], out_labels[4] == predicted_label, recent_preds.count(4)),
              # (preds[0][2], -20, 20, out_labels[2], out_labels[2] == predicted_label, recent_preds.count(2)),
              ((sr_scale_factor * data.shape[1] / target_sr)*-1.0, 0, -10, "Latency (s)", False, 0),
            ], 10, 100)
          preds_since_print = 0
      
      # for removing from original data
      samples_completed = (data_predict.shape[1] - window_samples - leftover_samples)
      sample_start_offset += samples_completed
      slice_start = floor(samples_completed / sr_scale_factor)
      del data_predict
      del data_stride
      # now remove data from original data
      data = data[:, slice_start:]
    # sleep at least long enough for next data chunk
    sleep(sleep_more + (sliding_step_samples / target_sr))
    # DO THINGS
  except KeyboardInterrupt:
    # write predictions to file
    if log_results:
      predictions.to_csv(prediction_file, index=False)
    # nicely close EEG connection
    eeg_source.stop()
    sys.exit()
