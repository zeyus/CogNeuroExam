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
