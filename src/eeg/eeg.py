import time
from typing import List, Tuple
from nptyping import NDArray, Float64
import numpy as np
from scipy.signal import savgol_filter
import sounddevice as sd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, NoiseTypes
import pyxdf
import pandas as pd

class EEG(object):
  def __init__(self, dummyBoard: bool = False) -> None:
    self.params = BrainFlowInputParams()
    self.curves = []
    if(dummyBoard):
      self._prepare_dummy_board()
    else:
      self._prepare_board()
    self.exg_channels = BoardShim.get_exg_channels(self.board_id)
    self.accel_channels = BoardShim.get_accel_channels(self.board_id)
    self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
    self.window_size = 4
    self.num_points = self.window_size * self.sampling_rate

    # self.start_stream()
    
  def start_stream(self) -> None:
    self.board.prepare_session()
    self.board.start_stream()

  def _send_command(self, command: str) -> None:
    response = self.board.config_board(command)
    return response

  def _set_channels_to_defaults(self) -> None:
    self._send_command(b'd')

  def _channel_config(self,
                        channel: int,
                        disable: bool = False,
                        gain: int = 6,
                        input_type: int = 0,
                        bias: bool = True,
                        srb1: bool = False,
                        srb2: bool = True) -> bool:
    """
    Change SRB1, SRB2, BIAS
    NOTE: this will reset the channels amp settings
    """
    if channel < 1 or channel > self.exg_channels:
      raise ValueError('Invalid channel number')
    if gain < 0 or gain > 6:
      raise ValueError('Invalid gain value')
    if input_type < 0 or input_type > 7:
      raise ValueError('Invalid input type')

    self._send_command('x {channel} {disable} {gain} {input_type} {bias} {srb1} {srb2} X'.format(
      channel = channel,
      disable = int(disable),
      gain = gain,
      input_type = input_type,
      bias = int(bias),
      srb1 = int(srb1),
      srb2 = int(srb2)
    ))
    return True

  def _start_sd_recording(self) -> bool:
    self._send_command('J')
    return True

  def _stop_sd_recording(self) -> bool:
    self._send_command('j')
    return True

  def _set_sample_rate(self, sample_rate: int = 6) -> bool:
    """
    0 = 16000 Hz
    1 = 8000 Hz
    2 = 4000 Hz
    3 = 2000 Hz
    4 = 1000 Hz
    5 = 500 Hz
    6 = 250 Hz
    """
    if sample_rate < 0 or sample_rate > 6:
      raise ValueError('Invalid sample rate switch')
    self._send_command('~{}'.format(sample_rate))
    return True
  
  def _start_time_stamping(self) -> bool:
    self._send_command('<')
    return True

  def _stop_time_stamping(self) -> bool:
    self._send_command('>')
    return True
    
  def _prepare_board(self) -> None:
    self.params.serial_port = 'COM3'
    self.board_id = 2  # cyton daisy
    self.update_speed_ms = 50
    self.board = BoardShim(self.board_id, self.params)

  def _prepare_dummy_board(self) -> None:
    self.board_id = BoardIds.SYNTHETIC_BOARD.value
    self.board = BoardShim(self.board_id, self.params)
    self.update_speed_ms = 50

  """Pull latest data from ringbuffer."""
  def poll(self, clear = True) -> NDArray[Float64]:
    if clear:
      return self.board.get_board_data(self.num_points)
    else:
      return self.board.get_current_board_data(self.num_points)
    
  def tag(self, tag:float) -> None:
    self.board.insert_marker(tag)

  def stop(self) -> None:
    if self.board.is_prepared():
      self.board.stop_stream()
      self.board.release_session()
      
class Filtering(object):
  def __init__(self, exg_channels: List[int], sampling_rate: int) -> None:
    self.exg_channels = exg_channels
    self.sampling_rate = sampling_rate

  def butterworth_lowpass(self, data: NDArray[Float64], cutoff = 49.0) -> NDArray[Float64]:
    for _, channel in enumerate(self.exg_channels):
      # DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
      DataFilter.perform_lowpass(data[channel], self.sampling_rate, cutoff, 1,
          FilterTypes.BUTTERWORTH.value, 0)
    return data

  def filter_50hz(self, data: NDArray[Float64]) -> NDArray[Float64]:
    for _, channel in enumerate(self.exg_channels):
      DataFilter.remove_environmental_noise(data[channel], self.sampling_rate, NoiseTypes.FIFTY.value)
    return data

class Audio(object):
  middle_c: float = 261.63
  pcm_sr: int = 44100
  attenuate: float = 0.2

  def scale_eeg_to_pcm_amp(x: NDArray[Float64], out_range=(-32767, 32767)) -> NDArray[Float64]:
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
  
  def resample(x: NDArray[Float64], sr_in: int, sr_out: int = None) -> NDArray[Float64]:
    if sr_out is None:
      sr_out = Audio.pcm_sr
    return np.interp(np.arange(0, len(x), sr_in / sr_out), np.arange(0, len(x)), x)

  def play(x: NDArray[Float64]) -> None:
    sd.play(x * Audio.attenuate, Audio.pcm_sr)
    time.sleep(len(x) / Audio.pcm_sr)
  
  def smooth(x: NDArray[Float64]) -> NDArray[Float64]:
    return np.convolve(x, np.ones(5), 'same') / 5

  def filter_savitzky_golay(x: NDArray[Float64], window_size: int = 5, order: int = 2) -> NDArray[Float64]:
    return savgol_filter(x, window_size, order)

class EEGReader(object):
  def parse_obci_header(file: str) -> Tuple[dict, int]:
    skip = 0
    headers = {}
    with open(file, 'rt') as f:
      for line in f:
        if not line.startswith("%"):
          break
        skip += 1
        if line.startswith("%Number of channels"):
          headers["exg_channels"] = int(line.split("=")[1])
        elif line.startswith("%Sample Rate"):
          headers["sampling_rate"] = int(line.split("=")[1][:-3]) # remove " Hz"
        elif line.startswith("%Board"):
          headers["board"] = line.split("=")[1].strip()
    return headers, skip

  def read_openbci_txt(file: str) -> Tuple[pd.DataFrame, dict]:
    headers, skip = EEGReader.parse_obci_header(file)
    return pd.read_csv(file, sep=',', header=skip), headers

  def read_xdf(file: str) -> Tuple[List[dict], dict]:
    return pyxdf.load_xdf(file)

class EEGWriter(object):
  def write_xdf(file: str, data: List[dict], headers: dict) -> None:
    pyxdf.save_xdf(file, data, headers)
