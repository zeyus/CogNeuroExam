# CogNeuroExam


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