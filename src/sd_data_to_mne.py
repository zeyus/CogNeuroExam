import os
import mne
import numpy as np


data_file = 'data/OBCI_10.TXT'
data_file_clean = 'data/OBCI_10_clean.TXT'


sfreq = 1000 # Hz
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'emg', 'eeg', 'eeg', 'stim']
ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'EMG1', 'P3', 'P4', 'marker']

ADS1299_Vref = 4.5
ADS1299_gain = 24.0

eeg_scale = (ADS1299_Vref / float ((pow(2, 23) - 1)) / ADS1299_gain * 1000000.)
accel_scale = (0.002 / (pow (2, 4)))


# check if clean data file exists
if not os.path.isfile(data_file_clean):
  print('Clean data file needs to be created')
  with open(data_file, 'rb') as f:
    data = f.read()
    # remove all trailling hex FF bytes
    data = data.rstrip(b'\xff')
    # save as new file
    with open(data_file_clean, 'wb') as f:
      f.write(data)
    del data

def hex_to_eeg_scaled_float(hexval):
  """
  Convert hex value to signed 24 bit number
  """
  # convert hex to int
  hexval = int(hexval, 16)
  # convert to signed int
  if hexval > 0x7fffff:
    hexval = hexval - 0x1000000
  return hexval * eeg_scale / 1000000. # the / 1000000 is for MNE

def hex_to_signed_16_bit_char(hexval):
  """
  Convert hex value to signed 16 bit number
  """
  # convert hex to int
  hexval = int(hexval, 16)
  # convert to signed int
  if hexval > 0x7fff:
    hexval = hexval - 0x10000
  return float(hexval)

def hex_to_signed_8_bit_number(hexval):
  """
  Convert hex value to signed 8 bit number
  """
  # convert hex to int
  hexval = int(hexval, 16)
  # convert to signed int
  if hexval > 0x7f:
    hexval = hexval - 0x100
  return hexval
  

eeg_data = np.genfromtxt(data_file_clean, comments='%', delimiter=',', skip_header=2,
  usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17),
  dtype=np.float64,
  converters={
    # 0: hex_to_signed_8_bit_number,
    1: hex_to_eeg_scaled_float,
    2: hex_to_eeg_scaled_float,
    3: hex_to_eeg_scaled_float,
    4: hex_to_eeg_scaled_float,
    5: hex_to_eeg_scaled_float,
    6: hex_to_eeg_scaled_float,
    7: hex_to_eeg_scaled_float,
    8: hex_to_eeg_scaled_float,
    9: hex_to_eeg_scaled_float,
    10: hex_to_eeg_scaled_float,
    11: hex_to_eeg_scaled_float,
    12: hex_to_eeg_scaled_float,
    13: hex_to_eeg_scaled_float,
    14: hex_to_eeg_scaled_float,
    15: hex_to_eeg_scaled_float,
    16: hex_to_eeg_scaled_float,
    17: hex_to_signed_16_bit_char,
    # 18: hex_to_signed_16_bit_char,
    # 19: hex_to_signed_16_bit_char
  })


print(eeg_data.shape)
eeg_data = eeg_data.T
print(eeg_data.shape)


info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(eeg_data, info)

print("Saving raw data to disk...{}".format(data_file))
# save as mne-python format
raw.save(data_file.replace('.TXT', '_raw.fif'), overwrite=True)

