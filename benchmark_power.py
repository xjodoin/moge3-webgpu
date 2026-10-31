"""Log macOS power/thermal state every ten seconds until a stop file appears."""
import json
from pathlib import Path
import subprocess
import sys
import time

stop = Path(sys.argv[1])
if stop.exists():
    raise SystemExit('Choose a new stop path or remove the previous stop file first')
try:
    while not stop.exists():
        battery = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        thermal = subprocess.check_output(['pmset', '-g', 'therm'], text=True)
        print(json.dumps({'time': time.time(), 'ac': "Now drawing from 'AC Power'" in battery,
                          'battery': battery.strip(), 'thermal': thermal.strip()}), flush=True)
        for _ in range(10):
            if stop.exists():
                break
            time.sleep(1)
except KeyboardInterrupt:
    pass
