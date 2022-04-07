"""
Configuration for data structure and saving
"""

# Where to save the participant data
DATA_PATH = './data'
LOG_FORMAT = 'logfile_{timestamp}_{id}_{condition}.csv'
EEG_FORMAT = 'eeg_{timestamp}_{id}_{condition}.RAW'

# Data columns
COLS = [
    # Timestamp YYYY-MM-DD HH:MM:SS
    'timestamp',
    # Participant ID
    'id',
    # Current word
    'word',
    # Display time
    'time',
    # Which condition: (control, experimental)
    'condition',
    # Integer: The nth word displayed of the story
    'sequence',
]


# 14 eeg, 2 emg and 3 accel channels
BCI_CHANNEL_TYPES = ['misc'] + ['eeg'] * 14 + ['emg'] + ['eeg'] + ['misc'] * 3 + ['stim']

# Channel names orig: ["Fp1,Fp2,C3,C4,P7,P8,O1,O2,F7,F8,F3,F4,T7,T8,P3,P4"]
BCI_CHANNEL_NAMES = ["pkg","Fp1","Fp2","C3","C4","P7","P8","O1","O2","F7","F8","F3","F4","T7","EMG1","P3","P4","AX","AY","AZ","stim"]

# mainly to slice the data
BCI_CHANNEL_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 31]

ENABLE_FIFTY_HZ_FILTER = False