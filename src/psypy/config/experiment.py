"""
Specific configuration for the experiment / interface.
"""

N_TRIALS = 30

CONDS = {
  'left': 'left',
  'right': 'right',
  'neutral': 'neutral',
}

PROMPTS = {
    'left': 'res/stimuli/arrow_left.png',
    'right': 'res/stimuli/arrow_right.png',
    'neutral': 'res/stimuli/arrow_up.png',
    'fixation': 'res/whitecircle.png',
    'ready': 'res/redcircle.png',
}

EEG_TAGS = {
    'left': 'l',
    'right': 'r',
    'neutral': 'n',
    'fixation': 'f',
    'ready': 'k',
}

# Informational messages for the participants
MESSAGES = {
    # Initial instructions before the experiment starts
    'instructions': """
        In this experiment you will be asked to kinesthetically imagine a fist clench, of either right or left hand.  \n
Before every round a cross is presented on the screen, for 2 seconds. Please focus on the cross. \n
After the cross disappears, you will see a red circle, indicating that in 1 second, an arrow, pointing either left, right or forward will appear. \n
If the arrow is pointing right, you have to imagine clenching your right first. If left - left fist.  \n
The arrow will disappear after 4 seconds. During these 4 seconds, please imagine clenching your fist continously. \n
This means clenching your fist and continuing to clench for the whole time, not clenching it as many times as possible. \n
If the arrow points forward, simply look at the screen, don't deliberately think of anything.  \n
Please keep your eyes open while imagining movements and try to move as little as possible.  \n
Press any key to continue.

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
