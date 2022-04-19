"""
Specific configuration for the experiment / interface.
"""

N_TRIALS = 30

CONDS = {
  'clench': 'clench',
  'right': 'right',
  'neutral': 'neutral',
}

PROMPTS = {
    'clench': 'res/stimuli/clench.jpg',
    'right': 'res/stimuli/arrow_right.png',
    'neutral': 'res/stimuli/arrow_up.png',
    'fixation': 'res/whitecircle.png',
    'ready': 'res/redcircle.png',
}

EEG_TAGS = {
    'clench': 'l',
    'right': 'r',
    'neutral': 'n',
    'fixation': 'f',
    'ready': 'k',
}

# Informational messages for the participants
MESSAGES = {
    # Initial instructions before the experiment starts
    'instructions': """
        Just do the thing...
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
        Break
        """,
    # Final message at the end of the experiment
    'complete': """
        Thank you for your participation.  \n
You can press any key to yeet out from the experiment. \n
        """
}
