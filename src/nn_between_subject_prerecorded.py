from datetime import datetime
from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet, EEGNet, EEGNetStrided, BENDRClassifier
from dn3.data.utils import get_dataset_max_and_min
from mne import set_config as mne_set_config
from nn.common import custom_raw_loader, write_model_results
import torch

experiment = ExperimentConfig("./src/dn3_between_subject_conf.yml")

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
    nnm = EEGNetStrided.from_dataset(dataset, **experiment.model_args.as_dict())
    model_name = type(nnm).__name__
    return StandardClassification(nnm, cuda=experiment.use_gpu, **experiment.classifier_args.as_dict())

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
    best_model = process.save_best()
    torch.save(best_model, "trained_models/{}_{}_{}.pt".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), model_name, subject_name))
    

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
    
