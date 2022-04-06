"""
Main word-by-word experiment script
"""

import sys
import os
import glob
import random
import threading
import time
from typing import List
import pandas as pd
import numpy as np
from eeg.eeg import EEG, Filtering

# Add src directory to path for module loading
sys.path.append(os.path.abspath(os.path.dirname(
    os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep))

#pylint: disable=wrong-import-position
import config
#pylint: enable=wrong-import-position

def setup():
    """
    Prepare psychopy environment and settings
    """

    #pylint: disable=import-outside-toplevel
    from psychopy import prefs
    #pylint: enable=import-outside-toplevel
    prefs.hardware['audioLib'] = config.psy.PREFS.get('audioLib')
    prefs.general['winType'] = config.psy.PREFS.get('winType')
    prefs.hardware['highDPI'] = config.psy.PREFS.get('highDPI')
    prefs.saveUserPrefs()
    #pylint: disable=import-outside-toplevel
    import app.psy as psy
    #pylint: enable=import-outside-toplevel

    # Ensure data directory exists and is writeable
    if not os.path.exists(config.data.DATA_PATH):
        try:
            os.mkdir(config.data.DATA_PATH)
        except Exception as ex:
            raise SystemError(
                'Cannot write to {}'.format(os.path.abspath(config.data.DATA_PATH))) from ex
    return psy.Psypy(config.psy.PREFS)

def get_sequence(nreps = 2) -> dict:
    """
    Prepare stimuli sequence for the experiment.
    """

    conditions = config.exp.CONDS

    sequence = []
    for cond in conditions:
        sequence = sequence + [cond] * nreps

    # shuffle sequence
    random.shuffle(sequence)

    return sequence

def write_experiment_data(timing_data: List[dict], participant: list, condition: str):
    """
    Save the experimental results
    """

    # Prepare dataframe using configured columns
    d_f = pd.DataFrame(columns=config.data.COLS)

    # Loop through the results for this participant
    for row in timing_data:
        d_f = d_f.append({
            'timestamp': row.get('timestamp'),
            'id': participant[0],
            'word': row.get('word'),
            'time': row.get('time'),
            'condition': condition.get('condition'),
            'sequence': row.get('sequence'),
        }, ignore_index = True)
    # Write the data to a csv file in the data directory
    d_f.to_csv(
        config.data.DATA_PATH + os.path.sep +
        config.data.LOG_FORMAT.format(
            timestamp = time.strftime('%Y-%m-%d_%H_%M_%S'),
            id=participant[0],
            condition=condition.get('condition')))


def collect_cont(streamer: EEG, stop_event: threading.Event, ready_event: threading.Event, participant: list):
    """
    Collect brain data
    """

    # get data file name
    file_out = str(
        config.data.DATA_PATH +
        os.path.sep +
        config.data.EEG_FORMAT.format(
            timestamp = time.strftime('%Y-%m-%d_%H_%M_%S'),
            id=participant[0],
            condition='experiment'))
    
    # start collecting data from board
    streamer.start_stream()
    # MNE data
    sr = streamer.sampling_rate
    ch_types = config.data.BCI_CHANNEL_TYPES
    ch_names = config.data.BCI_CHANNEL_NAMES
    ch_idx = config.data.BCI_CHANNEL_INDEXES

    filter_func = None
    if config.data.ENABLE_FIFTY_HZ_FILTER:
        streamFilter = Filtering(streamer.exg_channels, sr)
        filter_func = streamFilter.filter_50hz

    # save metadata (i mean it's not necessary but it's nice in case someone else wants to use it)
    with open(file_out + '.header', 'w') as f:
        f.write('Sampling rate: {}\n'.format(sr))
        f.write('Channel types: {}\n'.format(ch_types))
        f.write('Channel names: {}\n'.format(ch_names))

    with open(file_out, 'w') as f:
        # trigger ready signal for main thread to start experiment
        f.write(','.join(ch_names) + '\n')
        ready_event.set()

        # stop after recieving stop signal from main thread (experiment is over)
        while not stop_event.is_set():
            # sleep first for data in buffer
            time.sleep(1)
            # get data from buffer
            bci_data = streamer.poll()
            if bci_data is not None:
                if filter_func is not None:
                    bci_data = filter_func(bci_data)
                # write data to file, with channels as columns, rows as samples
                np.savetxt(f, bci_data[ch_idx,:].T, delimiter=',')
    # shutdown board's data stream
    if streamer is not None:
        streamer.stop()
    
    # trigger ready event to let main thread know data has been saved
    ready_event.set()


def run_experiment():
    """
    Step by step script for the experiment
    """
    #pylint: disable=import-outside-toplevel
    import app.psyparticipant as psyp
    #pylint: enable=import-outside-toplevel
    

    # Ask for participant ID
    participant = psyp.display_participant_dialogue()

    # Prepare psychopy
    psypy = setup()
    
    from psypy.app.psy import shutdown_psychopy

    # Get the participants condition
    sequence = get_sequence(config.exp.N_TRIALS)

    psypy.hide_cursor()
    # Show instructions
    psypy.display_text_message(config.exp.MESSAGES.get('instructions'))
    # show break message while waiting for EEG connection
    psypy.display_text_message(config.exp.MESSAGES.get('wait'), wait = False)
    # set up events for communicating with threads
    stop_event = threading.Event()
    ready_event = threading.Event()
     # start recording brain data
    streamer = EEG(dummyBoard=False)
    thread_cont = threading.Thread(target = collect_cont, args = [streamer, stop_event, ready_event, participant])
    thread_cont.start()
    # wait for eeg recording to start
    while not ready_event.is_set():
        time.sleep(0.1)

    for stim in sequence:
        psypy.call_on_next_flip(streamer.tag, config.exp.EEG_TAGS.get('fixation'))
        psypy.display_image(config.exp.PROMPTS.get('fixation'), 1)
        psypy.call_on_next_flip(streamer.tag, config.exp.EEG_TAGS.get('ready'))
        psypy.display_image(config.exp.PROMPTS.get('ready'), 2)
        psypy.call_on_next_flip(streamer.tag, config.exp.EEG_TAGS.get(stim))
        psypy.display_image(config.exp.PROMPTS.get(stim), 4)
        psypy.display_text_message(config.exp.MESSAGES.get('wait'), wait = False, wait_time = 1)
    # We can't accept input until the data saves
    psypy.display_text_message(config.exp.MESSAGES.get('wait'), wait = False)
    # Save the data
    #write_experiment_data(timing_data, participant, condition)
    # finish up EEG recording
    streamer.tag(5)
    ready_event.clear()
    # let thread know to stop recording
    stop_event.set()
    # wait for thread to finish
    while not ready_event.is_set():
        time.sleep(0.1)
    # Byeeeeeeeee
    psypy.show_cursor()
    psypy.display_text_message(config.exp.MESSAGES.get('complete'))

    # cleanup, garbage collection, etc
    del psypy
    shutdown_psychopy()
    
