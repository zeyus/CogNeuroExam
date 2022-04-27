from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet
from dn3.data.utils import get_dataset_max_and_min


experiment = ExperimentConfig("./src/dn3_conf.yml")
ds_config = experiment.datasets['clencher']

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

def make_model_and_process():
    tidnet = TIDNet.from_dataset(dataset)
    return StandardClassification(tidnet, cuda=experiment.use_gpu, learning_rate=ds_config.lr, **experiment.architecture.as_dict())

# results = list()
for subject_name in dataset.get_thinkers():
    if not subject_name == 'l':
        continue
    print("Processiong subject: {}".format(subject_name))
    thinker = dataset.thinkers[subject_name]
    # note, "testing" is used for validation, as you can't set test fraction to 0.
    training, _, testing = thinker.split(test_frac = ds_config.train_params.validation_fraction, validation_frac = 0)
    process = make_model_and_process()
    process.fit(training_dataset=training, validation_dataset=testing, **ds_config.training_configuration.as_dict())
    

# SPLIT BY SUBJECT, NOT IDEAL
# for training, validation, test in dataset.lmso(ds_config.train_params.folds):
#     process = make_model_and_process()

#     process.fit(training_dataset=training, validation_dataset=validation, **ds_config.training_configuration.as_dict())

#     results.append(process.evaluate(test)['Accuracy'])

# print(results)
# print("Average accuracy: {:.2%}".format(sum(results)/len(results)))

