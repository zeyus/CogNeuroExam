from mne.io import read_raw_fif, Raw
from mne import rename_channels
from mne.channels import make_standard_montage
from mne import set_config as mne_set_config
from pathlib import Path
from datetime import datetime
from dn3.trainable.models import EEGNetStrided


mne_set_config('MNE_STIM_CHANNEL', 'STI101')

channel_rename_map = {
    'marker': 'STI101'
}

# custom loader to allow setting montage
def custom_raw_loader(fname: Path, preload=True) -> Raw:
    raw = read_raw_fif(fname, preload=False, verbose=False)
    rename_channels(raw.info, channel_rename_map)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False, on_missing='ignore')
    if preload:
        raw.load_data()
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