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
