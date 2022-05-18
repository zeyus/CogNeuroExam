# CogNeuroExam



#### Data
OBCI_13 -> Luke, 2022-05-2? (??/r imagine)
OBCI_14 -> Alex, 2022-05-22 (??/r imagine)
OBCI_15 -> Luke, 2022-05-2? (??/r imagine)
OBCI_16 -> Alex, 2022-05-2? (??/r imagine)
OBCI_17 -> ??? (??/r imagine)
OBCI_18 -> Renee, 2022-05-2? (clench/r imagine)
OBCI_19 -> Alex, 2022-04-27 (l/r imagine)
OBCI_1A -> Alex, 2022-04-28 (l/r imagine)
OBCI_1E -> Luke, 2022-04-28 (l/r imagine)
OBCI_1F -> Alex, 2022-05-05 (l/r imagine)
OBCI_20 -> Luke, 2022-05-05 (l/r imagine)
OBCI_21 -> Luke, 2022-05-05 (l/r imagine)
OBCI_22 -> Alex, 2022-05-05 (l/r imagine)



#### Reqs

Experiment : conda env conda_env_data_collection.yml , essentially psychopy with some EEG tools
Data analysis / format conversion : conda_env_analysis.yml (maybe not up to date), MNE ...
Model training: pytorch, MNE, dn3 (zeyus fork), ...
Model deployment: TBC (onnx, pytorch...)



#### Other stuff


https://openbci.com/forum/index.php?p=/discussion/2461/questions-on-cyton-srb1-srb2-bias-and-the-openbci-electrode-cap


https://docs.openbci.com/Cyton/CytonSDK/

https://docs.openbci.com/Cyton/CytonSDCard/


- options, write to SD card, get higher sample rate

e.g. disable SRB1 and SRB2 on channel 14

example flow: (sent as individual bytes)

`d` To set all channels to default

set channel 14 to no SRB1 and no SRB2
`x 14 0 6 0 0 0 0 X`

set sample rate:
`~6` # 250 hz
`~0` # 16000 hz
`~4` # 2000 hz

start SD card logging:
`F` # 30min
`G` # 1hr

start time stamping:
`<`

stop timestamping:
`>`

stop SD card logging:
`j`


ONNX RUNTIME
`build.bat --skip_submodule_sync --use_cuda --cuda_version=11.5 --cuda_home="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5" --cudnn_home="C:/Program Files/NVIDIA/CUDNN/v8.3" --enable_memory_profile --parallel=8 --enable_cuda_profiling --config=Release`



