from mne.io import read_raw_fif, Raw
from mne import rename_channels, merge_events, find_events
from mne.channels import make_standard_montage
from mne import set_config as mne_set_config
from pathlib import Path
from datetime import datetime
from dn3.configuratron import DatasetConfig
from dn3.trainable.models import EEGNetStrided, EEGNet
import numpy as np


mne_set_config('MNE_STIM_CHANNEL', 'STI101')

channel_rename_map = {
    'marker': 'EX1'
}

# custom loader to allow setting montage
def custom_raw_loader(ds_conf: DatasetConfig,fname: Path) -> Raw:
    raw = read_raw_fif(fname, preload=False, verbose=False)
    rename_channels(raw.info, channel_rename_map)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False, on_missing='ignore')
    raw.load_data()
    if not hasattr(ds_conf, 'event_prep') or ds_conf.event_prep.do_not_run:
        return raw
    
    off_events = ds_conf.event_prep.off_events
    # combine_events = [110, 111, 113]
    combine_events = ds_conf.event_prep.combine_events
    combined_id = ds_conf.event_prep.combined_event_id
    # offset "off events"
    move_off_events_ms = ds_conf.event_prep.move_off_events_ms

    events = find_events(raw, stim_channel='EX1')
    if not move_off_events_ms == 0:
        event_ids = events[:, 2]
        event_ids_idx = np.argwhere((event_ids == off_events[0]) | (event_ids == off_events[1]))
        events[:, 0][event_ids_idx] = events[:, 0][event_ids_idx] + move_off_events_ms
        events[:, 0] = events[np.argsort(events[:, 0]), 0]
    # create a singular combined event if required event
    if len(combine_events) > 0:
        events = merge_events(events, ids=combine_events, new_id=combined_id, replace_events=True)
    raw.add_events(events, stim_channel='EX1', replace=True)
    return raw


def write_model_results(model_info, results):
    curdate = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = 'results_{date}.csv'.format(date=curdate)
    with open('output/{}'.format(filename), 'w') as f:
        f.write(f'{model_info}\n\n\n\n')
        for subject, train_log, valid_log in results:
            train_log.to_csv('output/details_{subject}_{date}_train.csv'.format(date=curdate, subject=subject), header=True)
            valid_log.to_csv('output/details_{subject}_{date}_valid.csv'.format(date=curdate, subject=subject), header=True)
            f.write(f'Subject: {subject}\n\n')
            best_loss = min(train_log['loss'])
            best_accuracy = max(train_log['Accuracy'])
            f.write('Training:\n')
            f.write(f'Best loss: {best_loss}\n')
            f.write(f'Best accuracy: {best_accuracy}\n')
            f.write('Validation:\n')
            best_loss = min(valid_log['loss'])
            best_accuracy = max(valid_log['Accuracy'])
            f.write(f'Best loss: {best_loss}\n')
            f.write(f'Best accuracy: {best_accuracy}\n')
            f.write('\n')

class EEGNetStridedOnnxCompat(EEGNetStrided):
    # def __init__(self, *args, **kwargs):
    #     super(EEGNetStrided, self).__init__(*args, **kwargs)

    def features_forward(self, x, *args, **kwargs):
        return super().features_forward(x)

    # @classmethod
    # def from_dataset(cls, *args, **kwargs):
    #     return super(EEGNetStrided, cls).from_dataset(*args, **kwargs)


class EEGNetOnnxCompat(EEGNet):
    # def __init__(self, *args, **kwargs):
    #     super(EEGNetStrided, self).__init__(*args, **kwargs)

    def features_forward(self, x, *args, **kwargs):
        return super().features_forward(x)

    # @classmethod
    # def from_dataset(cls, *args, **kwargs):
    #     return super(EEGNetStrided, cls).from_dataset(*args, **kwargs)