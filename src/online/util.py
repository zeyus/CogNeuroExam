"""
Classes for loading and preparing an onnx model for online classification.
"""

from typing import List
import numpy as np
import onnxruntime as rt
from onnxruntime.capi._pybind_state import SessionOptions
import logging
from dn3.transforms.instance import MappingDeep1010
from mne.io.constants import FIFF
import torch


class OnnxOnline:
  def __init__(self, model_path: str, sess_opts: SessionOptions = None):
    if sess_opts is None:
      sess_opts = self._default_sess_opts()
    self.session = rt.InferenceSession(model_path, sess_options=sess_opts)
    self.inputs = self.session.get_inputs()
    self.outputs = self.session.get_outputs()
    logging.info(f"Loaded model {model_path}")
    logging.info(f"Model input [{self.inputs[0].name}] shape: {self.inputs[0].shape}")
    logging.info(f"Model output [{self.outputs[0].name}] shape: {self.outputs[0].shape}")
    
  def _default_sess_opts(self) -> SessionOptions:
    sess_opts = rt.SessionOptions()
    sess_opts.intra_op_num_threads = 8
    sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_opts.enable_profiling = True
    return sess_opts

  def predict(self, x):
    """
    Predict the class of the input x.
    """
    return self.session.run(None, {'X': x})[0]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class DN3D1010(MappingDeep1010):

  def __init__(self, ch_names: List, ch_types: List, use_ch_idx: List, dmin: float = -0.3, dmax: float = 0.3):
    fake_dataset = dotdict()
    fake_dataset.channels = np.array([np.array([x, self._mne_map(y)]) for x, y in zip(ch_names, ch_types)])
    fake_dataset.info = dotdict()
    fake_dataset.info.data_min = dmin
    fake_dataset.info.data_max = dmax
    self.use_ch_idx = use_ch_idx
    self.ch_types = ch_types
    super().__init__(fake_dataset)
  
  def __call__(self, x):
    """
    Transform the input data.
    """
    # Turn the input ndarray into a tensor



    x = super().__call__(torch.from_numpy(x))
    return x

  def _mne_map(self, x: str) -> int:
    if x == 'eeg':
      x_type = FIFF.FIFFV_EEG_CH
    elif x == 'emg':
      x_type = FIFF.FIFFV_EMG_CH
    elif x == 'stim':
      x_type = FIFF.FIFFV_STIM_CH
    elif x == 'eog':
      x_type = FIFF.FIFFV_EOG_CH
    elif x == 'ecg':
      x_type = FIFF.FIFFV_ECG_CH
    elif x == 'misc':
      x_type = FIFF.FIFFV_MISC_CH
    else:
      raise ValueError(f"Unknown channel type: {x}")
    return int(x_type)

  def zero_discard(self, x: np.ndarray) -> np.ndarray:
    """
    Set channels that are not used to zero.
    """
    select = np.isin(range(x.shape[0]), self.use_ch_idx, invert=True)
    x[select, :] = 0
    return x
    

