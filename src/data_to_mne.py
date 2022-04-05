# this reads data saved from Brainflow / OpenBCI exg
# and converts it to mne-python format

import numpy as np
import mne

# lazy way using hardcoded values, we could extract them from the header
sfreq = 125
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'emg', 'eeg', 'eeg', 'stim']
ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'EMG1', 'P3', 'P4', 'marker']

data_file = 'data/eeg_2022-04-05_12_56_04_1_experiment.RAW'

# read csv
exg_data = np.loadtxt(data_file, delimiter=',', skiprows=1)
exg_data = exg_data.T
accel = exg_data[-5:-2]

print(accel.shape)
markers = exg_data[-1]
print(exg_data.shape)
exg_data = exg_data[1:17]
# uV to V
exg_data = exg_data / 1000000

print(markers.shape)
exg_data = np.append(exg_data, [markers], axis=0)
print(exg_data.shape)



info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(exg_data, info)

print("Saving raw data to disk...{}".format(data_file))
# save as mne-python format
raw.save(data_file.replace('.RAW', '_raw.fif'), overwrite=True)


