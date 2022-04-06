import os
import sys
emulator_dir = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..' + os.path.sep + 'contrib' + os.path.sep + 'emulator' + os.path.sep)
sys.path.append(emulator_dir)

from brainflow_emulator import cyton_windows

test_file = os.path.dirname(__file__) + os.path.sep + 'cyton_comms.py'

cyton_windows.main(['python', test_file, '--serial-port'])