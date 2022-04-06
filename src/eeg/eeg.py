from enum import Enum
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

class CytonSampleRate(Enum):
  SR_250 = 6
  SR_500 = 5
  SR_1000 = 4
  SR_2000 = 3
  SR_4000 = 2
  SR_8000 = 1
  SR_16000 = 0

class CytonInputType(Enum):
  ADSINPUT_NORMAL = 0
  ADSINPUT_SHORTED = 1
  ADSINPUT_BIAS_MEAS = 2
  ADSINPUT_MVDD = 3
  ADSINPUT_TEMP = 4
  ADSINPUT_TESTSIG = 5
  ADSINPUT_BIAS_DRP = 6
  ADSINPUT_BIAS_DRN = 7

class CytonRecordingDuration(Enum):
  MIN_5 = 'A'
  MIN_15 = 'S'
  MIN_30 = 'F'
  MIN_60 = 'G'
  MIN_120 = 'H'
  MIN_240 = 'J'
  MIN_720 = 'K'
  MIN_1440 = 'L'
  TEST_14_SEC = 'a'

class CytonChannel(Enum):
  CH_1 = 1
  CH_2 = 2
  CH_3 = 3
  CH_4 = 4
  CH_5 = 5
  CH_6 = 6
  CH_7 = 7
  CH_8 = 8
  CH_9 = 'Q'
  CH_10 = 'W'
  CH_11 = 'E'
  CH_12 = 'R'
  CH_13 = 'T'
  CH_14 = 'Y'
  CH_15 = 'U'
  CH_16 = 'I'

  @classmethod
  def from_channel_number(cls, channel_number: int) -> 'CytonChannel':
    return cls['CH_{}'.format(channel_number)]

class CytonGain(Enum):
  GAIN_1 = 0
  GAIN_2 = 1
  GAIN_4 = 2
  GAIN_6 = 3
  GAIN_8 = 4
  GAIN_12 = 5
  GAIN_24 = 6
  
class CytonCommands(Enum):
  SAMPLE_RATE_PREFIX = '~'
  BOARD_MODE_PREFIX = '/'
  CHANNEL_CONFIG_PREFIX = 'x'
  CHANNEL_CONFIG_SUFFIX = 'X'
  RESET_CHANNELS = 'd'
  SOFT_RESET_BOARD = 'v'
  QUERY_VERSION = 'V'
  QUERY_REGISTER = '?'
  SD_STOP = 'j'
  STREAM_START = 'b'
  STREAM_STOP = 's'
  TIMESTAMP_START = '<'
  TIMESTAMP_STOP = '>'
  CHANNEL_1_ON = '!'
  CHANNEL_1_OFF = '1'
  CHANNEL_2_ON = '@'
  CHANNEL_2_OFF = '2'
  CHANNEL_3_ON = '#'
  CHANNEL_3_OFF = '3'
  CHANNEL_4_ON = '$'
  CHANNEL_4_OFF = '4'
  CHANNEL_5_ON = '%'
  CHANNEL_5_OFF = '5'
  CHANNEL_6_ON = '^'
  CHANNEL_6_OFF = '6'
  CHANNEL_7_ON = '&'
  CHANNEL_7_OFF = '7'
  CHANNEL_8_ON = '*'
  CHANNEL_8_OFF = '8'
  CHANNEL_9_ON = 'Q'
  CHANNEL_9_OFF = 'q'
  CHANNEL_10_ON = 'W'
  CHANNEL_10_OFF = 'w'
  CHANNEL_11_ON = 'E'
  CHANNEL_11_OFF = 'e'
  CHANNEL_12_ON = 'R'
  CHANNEL_12_OFF = 'r'
  CHANNEL_13_ON = 'T'
  CHANNEL_13_OFF = 't'
  CHANNEL_14_ON = 'Y'
  CHANNEL_14_OFF = 'y'
  CHANNEL_15_ON = 'U'
  CHANNEL_15_OFF = 'u'
  CHANNEL_16_ON = 'I'
  CHANNEL_16_OFF = 'i'

  @classmethod
  def channel_number_on(cls, channel_number: int) -> 'CytonCommands':
    return cls['CHANNEL_{}_ON'.format(channel_number)]
  @classmethod
  def channel_number_off(cls, channel_number: int) -> 'CytonCommands':
    return cls['CHANNEL_{}_OFF'.format(channel_number)]

class CytonBoardMode(Enum):
  DEFAULT = 0
  DEBUG = 1
  ANALOG = 2
  DIGITAL = 3
  MARKER = 4
  GET_BOARD_MODE = '/'

class EEG(object):
  sdcard = False
  is_prepared = False
  def __init__(self, dummyBoard: bool = False, emg_channels: List[int] = [], serial_port = 'COM3') -> None:
    self.params = BrainFlowInputParams()
    self.serial_port = serial_port
    self.curves = []
    self.dummyBoard = dummyBoard
    if(dummyBoard):
      self._prepare_dummy_board()
    else:
      self._prepare_board()
    self.emg_channels = emg_channels
    self.exg_channels = BoardShim.get_exg_channels(self.board_id)
    self.accel_channels = BoardShim.get_accel_channels(self.board_id)
    self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
    self.window_size = 4
    self.num_points = self.window_size * self.sampling_rate

    # self.start_stream()
  
  def prepare(self):
    # connect to board
    self.board.prepare_session()
    self.is_prepared = True

  def start_stream(self, sdcard = True, sr: CytonSampleRate = None, duration: CytonRecordingDuration = CytonRecordingDuration.MIN_120) -> None:
    """
    Starts recording EEG data either to sd card or via dongle
    defaults to 120 minutes for SDCard recording
    Sample rate defaults to 250Hz for streaming over dongle
    and 1000Hz for recording sd card
    """
    
    if not self.is_prepared:
      self.prepare()
    # no channel config possible
    if self.dummyBoard:
      self.board.start_stream()
      return
    
    # set channels to default
    self._set_channels_to_defaults()
    # set EMG configuration
    self._config_emg_channels()

    # start streaming
    if not sdcard:
      if sr is None:
        sr = CytonSampleRate.SR_250
      self._set_sample_rate(sr)
      self.board.start_stream()
      return
    
    # sdcard
    self.sdcard = True
    if sr is None:
      sr = CytonSampleRate.SR_1000
      self._set_sample_rate(sr)

    # if we get here it's a real board
    # and sdcard is True
    self._start_sd_recording(duration)
  
  def _soft_reset(self) -> str:
    return self._send_command(CytonCommands.SOFT_RESET_BOARD.value)

  def _config_emg_channels(self) -> bool:
    """
    Disable SRB1 and 2 for EMG channels
    """
    for channel in self.emg_channels:
      ch = CytonChannel.from_channel_number(channel)
      self._channel_config(ch, disable=False, gain=CytonGain.GAIN_24, input_type=CytonInputType.ADSINPUT_NORMAL, bias=False, srb1=False, srb2=False)
    return True

  def disable_channel(self, channel: int) -> bool:
    """
    Disables a channel
    """
    ch = CytonCommands.channel_number_off(channel)
    return self._send_command(ch.value)

  def enable_channel(self, channel: int) -> bool:
    """
    Enables a channel
    """
    ch = CytonCommands.channel_number_on(channel)
    return self._send_command(ch.value)

  def _send_command(self, command: str) -> str:
    """
    Sends a command to the board
    """
    response = self.board.config_board(command)
    return response

  def _set_channels_to_defaults(self) -> None:
    """
    Sets all channels to default
    """
    self._send_command(CytonCommands.RESET_CHANNELS.value)

  def _channel_config(self,
                        channel: CytonChannel,
                        disable: bool = False,
                        gain: CytonGain = CytonGain.GAIN_24,
                        input_type: CytonInputType = CytonInputType.ADSINPUT_NORMAL,
                        bias: bool = True,
                        srb1: bool = False,
                        srb2: bool = True) -> bool:
    """
    Configure a single channel's settings
    Defaults:
      disable = False
      gain = GAIN_24
      input_type = ADSINPUT_NORMAL
      bias = True
      srb1 = False
      srb2 = True
    """

    self._send_command('{prefix}{channel}{disable}{gain}{input_type}{bias}{srb1}{srb2}{suffix}'.format(
      prefix = CytonCommands.CHANNEL_CONFIG_PREFIX.value,
      channel = channel.value,
      disable = int(disable),
      gain = gain.value,
      input_type = input_type.value,
      bias = int(bias),
      srb1 = int(srb1),
      srb2 = int(srb2),
      suffix = CytonCommands.CHANNEL_CONFIG_SUFFIX.value
    ))
    return True

  def _start_sd_recording(self, duration: CytonRecordingDuration = CytonRecordingDuration.MIN_120) -> bool:
    """
    Starts recording to sd card
    """
    self._send_command(duration.value)
    return True

  def _stop_sd_recording(self) -> bool:
    """
    Stops recording to sd card
    """
    self._send_command(CytonCommands.SD_STOP.value)
    return True

  def _set_sample_rate(self, sample_rate: CytonSampleRate) -> bool:
    """
    Sets the sample rate
    """
    self._send_command('{}{}'.format(CytonCommands.SAMPLE_RATE_PREFIX.value, sample_rate.value))
    return True
  
  def _start_time_stamping(self) -> bool:
    """
    Starts time stamping
    """
    self._send_command(CytonCommands.TIMESTAMP_START.value)
    return True

  def _stop_time_stamping(self) -> bool:
    """
    Stops time stamping
    """
    self._send_command(CytonCommands.TIMESTAMP_STOP.value)
    return True
    
  def _prepare_board(self) -> None:
    """
    Prepares the board for streaming / recording connection
    """
    self.params.serial_port = self.serial_port
    self.board_id = 2  # cyton daisy
    self.update_speed_ms = 50
    self.board = BoardShim(self.board_id, self.params)

  def _prepare_dummy_board(self) -> None:
    """
    Creates a dummy board
    """
    self.board_id = BoardIds.SYNTHETIC_BOARD.value
    self.board = BoardShim(self.board_id, self.params)
    self.update_speed_ms = 50

  def poll(self, clear = True) -> NDArray[Float64]:
    """
    Gets latest data from the board
    if clear is True, the ringbuffer is emptied
    """
    if clear:
      return self.board.get_board_data(self.num_points)
    else:
      return self.board.get_current_board_data(self.num_points)
    
  def tag(self, tag:float) -> None:
    """
    Tags the current sample with a float value
    """
    self.board.insert_marker(tag)

  def stop(self) -> None:
    """
    Stops streaming (if streaming) and disconnects board session
    """
    if self.board.is_prepared():
      if not self.sdcard:
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
