from dn3.configuratron.config import ExperimentConfig

experiment = ExperimentConfig("src/dn3_conf.yml")

dataset = experiment.datasets['public_dataset'].auto_construct_dataset()
# Do some awesome things