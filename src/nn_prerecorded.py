from dn3.configuratron import ExperimentConfig
from dn3.trainable.processes import StandardClassification
from dn3.trainable.models import TIDNet

experiment = ExperimentConfig("./src/dn3_conf.yml")
ds_config = experiment.datasets['clencher']

dataset = ds_config.auto_construct_dataset()


def make_model_and_process():
    tidnet = TIDNet.from_dataset(dataset)
    return StandardClassification(tidnet, cuda=experiment.use_gpu, learning_rate=ds_config.lr)

results = list()
for training, validation, test in dataset.lmso(ds_config.train_params.folds):
    process = make_model_and_process()

    process.fit(training_dataset=training, validation_dataset=validation, **ds_config.training_configuration.as_dict())

    results.append(process.evaluate(test)['Accuracy'])

print(results)
print("Average accuracy: {:.2%}".format(sum(results)/len(results)))

# results = dict()
# for training, validation, test in dataset.loso(ds_config.train_params.folds):
#     process = make_model_and_process()

#     process.fit(training_dataset=training, validation_dataset=validation,
#                 epochs=ds_config.training_configuration.epochs,
#                 batch_size=ds_config.training_configuration.batch_size)

#     for _, _, test_thinker in test.loso():
#         results[test_thinker.person_id] = process.evaluate(test_thinker)

# print(results)
# print("Average accuracy: {:.2%}".format(sum(results.values())/len(results)))