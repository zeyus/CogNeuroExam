
import sys
import os
import glob
import random
import threading
import time
from typing import List
import pandas as pd
import numpy as np
from eeg.eeg import EEG, Filtering, CytonSampleRate
# Add src directory to path for module loading
sys.path.append(os.path.abspath(os.path.dirname(
    os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep))

#pylint: disable=wrong-import-position
import config
#pylint: enable=wrong-import-position


def collect_cont(streamer: EEG, stop_event: threading.Event, ready_event: threading.Event, participant: list, max_dur_mins: int = 60):
    """
    Collect brain data
    """

    ch_types = config.data.BCI_CHANNEL_TYPES
    ch_names = config.data.BCI_CHANNEL_NAMES
    ch_idx = config.data.BCI_CHANNEL_INDEXES

    ch_emg = [index for index,value in enumerate(ch_types) if value == 'emg']
    streamer.emg_channels = ch_emg
    # start collecting data from board
    streamer.start_stream(sdcard=False, sr = CytonSampleRate.SR_250)
    # MNE data
    sr = streamer.sampling_rate

    ready_event.set()

    # stop after recieving stop signal from main thread (experiment is over)
    while not stop_event.is_set():
        # sleep first for data in buffer
        time.sleep(1/20)
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

def clASSify():

    # set up events for communicating with threads
    stop_event = threading.Event()
    ready_event = threading.Event()
    
    # start recording brain data
    streamer = EEG(dummyBoard=False)

    thread_cont = threading.Thread(target = collect_cont, args = [streamer, stop_event, ready_event])
    thread_cont.start()
    # wait for eeg recording to start
    while not ready_event.is_set():
        time.sleep(0.1)
    
    ready_event.clear()
    # let thread know to stop recording
    stop_event.set()
    # wait for thread to finish
    while not ready_event.is_set():
        time.sleep(0.1)
    # Byeeeeeeeee
