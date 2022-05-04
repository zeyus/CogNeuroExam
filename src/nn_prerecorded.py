from datetime import datetime
from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet, EEGNet, EEGNetStrided, BENDRClassifier
from dn3.data.utils import get_dataset_max_and_min
from mne.io import read_raw_fif, Raw
from mne import rename_channels
from mne.channels import make_standard_montage
from mne import set_config as mne_set_config
from pathlib import Path

channel_rename_map = {
    'marker': 'STI101'
}
mne_set_config('MNE_STIM_CHANNEL', 'STI101')

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
        


experiment = ExperimentConfig("./src/dn3_conf.yml")

if experiment.use_gpu:
    mne_set_config('MNE_USE_CUDA', 'True')

ds_config = experiment.datasets['clencher']


ds_config.add_custom_raw_loader(custom_raw_loader)

dataset = ds_config.auto_construct_dataset()
if isinstance(ds_config.data_min, bool) or isinstance(ds_config.data_max, bool):
    if experiment.experiment.get('deep1010', True):
        print("Cannot calculate data range with deep1010 enabled, please set to false, calculate then set to true again.")
        exit()
    dr = get_dataset_max_and_min(dataset)
    print("Calculated data range, please update dn3_conf.yml and rerun.")
    print(dr)
    exit()
# dataset.add_transform()
model_name = ''
def make_model_and_process():
    global model_name
    tidnet = EEGNetStrided.from_dataset(dataset, **experiment.model_args.as_dict())
    model_name = type(tidnet).__name__
    return StandardClassification(tidnet, cuda=experiment.use_gpu, **experiment.classifier_args.as_dict())

results = list()
for subject_name in dataset.get_thinkers():
    print("Processiong subject: {}".format(subject_name))
    thinker = dataset.thinkers[subject_name]
    # print(thinker.get_targets())
    # exit()
    # note, "testing" is used for validation, as you can't set test fraction to 0.
    training, _, testing = thinker.split(test_frac = ds_config.split_args.validation_fraction, validation_frac = 0)
    process = make_model_and_process()
    train_log, valid_log = process.fit(training_dataset=training, validation_dataset=testing, **experiment.fit_args.as_dict())
    results.append((subject_name, train_log, valid_log))

write_model_results({
    'model_type': model_name,
    'model_args': experiment.model_args.as_dict(),
    'classifier_args': experiment.classifier_args.as_dict(),
    'fit_args': experiment.fit_args.as_dict(),
    'split_args': ds_config.split_args.as_dict(),
    'data_min': ds_config.data_min,
    'data_max': ds_config.data_max,
    'hpf': ds_config.hpf,
    'lpf': ds_config.lpf,
    'tmin': ds_config.tmin,
    'tlen': ds_config.tlen,
    'use_avg_ref': ds_config.use_avg_ref,
    'notch_freq': ds_config.notch_freq
}, results)
    

# SPLIT BY SUBJECT, NOT IDEAL
# for training, validation, test in dataset.lmso(ds_config.train_params.folds):
#     process = make_model_and_process()

#     process.fit(training_dataset=training, validation_dataset=validation, **ds_config.training_configuration.as_dict())

#     results.append(process.evaluate(test)['Accuracy'])

# print(results)
# print("Average accuracy: {:.2%}".format(sum(results)/len(results)))

