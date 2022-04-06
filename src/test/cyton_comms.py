import os
import sys
import argparse
from time import sleep

src_dir = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..' + os.path.sep)
sys.path.append(src_dir)

from eeg.eeg import EEG, CytonCommands


parser = argparse.ArgumentParser()
parser.add_argument('--serial-port', type=str, help='serial port', required=True)
args = parser.parse_args()

streamer = EEG(dummyBoard=False, serial_port=args.serial_port)
streamer.prepare()
# get current sample rate
result = streamer._send_command(CytonCommands.QUERY_REGISTER.value)
print(result)
result = streamer._send_command(CytonCommands.SOFT_RESET_BOARD.value)
print(result)
streamer.start_stream(sdcard=False)
sleep(0.5)
streamer.stop()
# can't send commands that return binary response
# result = streamer._send_command(CytonCommands.STREAM_START.value)
# print(result)
# sleep(0.5)
# result = streamer._send_command(CytonCommands.STREAM_STOP.value)
# print(result)