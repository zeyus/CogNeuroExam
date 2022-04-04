"""
Specific configuration for the experiment / interface.
"""

CONDS = {
  'control': 'left',
  'experimental': 'right',
  'baseline': 'none',
}

PROMPTS = {
    'control': '''
        Imagine moving your left hand...
        ''',
    'experimental': '''
        Imagine moving your right hand...
        ''',
    'baseline': '''
        Do nothing...
        ''',
}

# Informational messages for the participants
MESSAGES = {
    # Initial instructions before the experiment starts
    'instructions': """
        In the following experiment you will be asked to imagine performing a movement while remaining as still as possible.

        """,
    # Break after the practice to prevent accidental start of the experiment
    'continue': """
        Press [space] to continue...
        """,
    # Displayed after the practice is complete
    'post_practice' : """
        The practice trial is now complete. Once you are ready, press [space] to begin the experiment.
        """,
    # Transient message while data is writing (hopefully very quick)
    'wait': """
        Please wait...
        """,
    # Final message at the end of the experiment
    'complete': """
        The experiment is over. Thank you for your participation!
        
        Have a wonderful day :)
        """
}

# 14 eeg, 2 emg and 3 accel channels
BCI_CHANNEL_TYPES = ['eeg'] * 14 + ['emg'] * 2 + ['AU'] * 3

# BCI_CHANNEL_NAMES = ["Fp1,Fp2,C3,C4,P7,P8,O1,O2,F7,F8,F3,F4,T7,T8,P3,P4"]
BCI_CHANNEL_NAMES = ["Fp1,Fp2,C3,C4,P7,P8,O1,O2,F7,F8,F3,F4,T7,T8,EMG1,EMG2,AX,AY,AZ"]

