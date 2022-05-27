from datetime import datetime
from time import sleep
from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet, EEGNet, EEGNetStrided, BENDRClassifier
from dn3.data.utils import get_dataset_max_and_min
from mne.io import read_raw_fif, Raw
from mne import rename_channels
from mne.channels import make_standard_montage
from mne import set_config as mne_set_config
from nn.common import custom_raw_loader, write_model_results, EEGNetStridedOnnxCompat, EEGNetOnnxCompat
from pathlib import Path

import torch

if __name__ == "__main__":


    experiment = ExperimentConfig("./src/nn_within_subject_conf.yml")

    if experiment.use_gpu:
        mne_set_config('MNE_USE_CUDA', 'True')
        device = 'cuda'

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
        nnm = EEGNetStridedOnnxCompat.from_dataset(dataset, **experiment.model_args.as_dict())
        model_name = type(nnm).__name__
        return StandardClassification(nnm, cuda=experiment.use_gpu, **experiment.classifier_args.as_dict())

    results = list()

    cur_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for subject_name in dataset.get_thinkers():
        if not subject_name == 'l':
            continue
        print("Processing subject: {}".format(subject_name))
        thinker = dataset.thinkers[subject_name]
        # print(thinker.get_targets())
        # exit()
        # note, "testing" is used for validation, as you can't set test fraction to 0.
        training, _, testing = thinker.split(test_frac = ds_config.split_args.validation_fraction, validation_frac = 0)
        process = make_model_and_process()
        train_log, valid_log = process.fit(training_dataset=training, validation_dataset=testing, **experiment.fit_args.as_dict())
        process.train(False)
        best_model = process.save_best()
        best_model_file = "trained_models/{}_{}_{}.pt".format(cur_date, model_name, subject_name)
        print("Saving model weights to: {}".format(best_model_file))

        torch.save(best_model, best_model_file)
        # some_data = [training[x] for x in range(0, experiment.fit_args.as_dict()['batch_size'], 1)]
        
        
        torch_model = process.classifier
        torch_model.eval()
        # m_kwa = experiment.model_args.as_dict()
        onnx_file = "trained_models/{}_{}_{}.onnx".format(cur_date, model_name, subject_name)
        print("Exporting ONNX model to: {}".format(onnx_file))
        torch.onnx.export(
            model=torch_model,
            # (batches, channels, n_samples)
            args=torch.randn(6, 90, round(ds_config.tlen * experiment.global_sfreq), device=device),
            f=onnx_file,
            export_params=True,
            opset_version=16, #15, # 16 is latest but deepsparse dev currently support <=15
            do_constant_folding=True,
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                        'output' : {0 : 'batch_size'}},
            verbose = True)

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
        'notch_freq': ds_config.notch_freq,
        'events': ds_config.events
    }, results)
        


