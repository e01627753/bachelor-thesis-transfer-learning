'''
Syntax for script execution:
python<optionally the version of your python> execute_models_on_tpu.py -m <model name of edge tpu compiled model>

EXAMPLE: python3 execute_models_on_tpu.py -m tl_model_8int_edgetpu
'''

########################################################################
######### import required libraries
########################################################################
import sys
import os
import argparse
import numpy as np
import time
from PIL import Image
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
'''
script has been taken from:
https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py
and adapted for the porpuse of this bachelor thesis. Other references:
https://coral.ai/docs/accelerator/get-started/#3-run-a-model-on-the-edge-tpu
'''

import pandas as pd
from pymeas.device import GPMDevice
from pymeas.output import CsvOutput
########################################################################
######### define ArgumentParser, add -b option and parse arguments
########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, help="Ein String als Option")
args = parser.parse_args()

print()
if not args.m:
    sys.exit("ERROR: You have to set -m as an option to clarify the model name of the model you want to execute!.")
else:
    model_name = args.m

########################################################################
######### load quantized tensorflow lite models
########################################################################
print("INFO: Loading 8 bit model:", model_name)
interpreter = make_interpreter("models/" + model_name + ".tflite")
interpreter.allocate_tensors()

########################################################################
######### load and reformat test data
########################################################################
size = common.input_size(interpreter)

image_files = os.listdir("data")  # list of image names in /data
num_images = len(image_files)

# initialize empty numpy array
X_input = np.empty((num_images, size[0], size[1], 3), dtype=np.float32)

for i, image_file in enumerate(image_files):
    # path to image
    image_path = os.path.join("data", image_file)
    
    # open and scale image in regards to interpreter
    image = Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS)

    # add image to numpy array
    X_input[i] = np.array(image)

print("X_input.shape:", X_input.shape)

########################################################################
######### connect to power meter
########################################################################
power_meter_device = GPMDevice(host="192.168.167.90")
power_meter_device.connect()

########################################################################
######### run inference and return speed measure
########################################################################
# quantize data
params = common.input_details(interpreter, 'quantization_parameters')
scale = params['scales']
zero_point = params['zero_points']
X_inf_test = (X_input / scale) + zero_point

duration = 0.0
# measure power consumption
measurement_thread = power_meter_device.start_power_capture()
for i in range(X_inf_test.shape[0]):
    
    common.set_input(interpreter, X_inf_test[i].astype(np.uint8))
    # run inference
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    duration += end_time-start_time
    print('%.1fms' % ((end_time-start_time) * 1000))

# stop measuring power consumption
power = power_meter_device.stop_power_capture(measurement_thread)
power_meter_device.disconnect()
data = [{'timestamp': key, 'value': value} for key, value in power.items()]
CsvOutput.save(f"power_measures/{model_name}.csv", field_names=['timestamp', 'value'], data=data)
df = pd.DataFrame(power.items(), columns=['timestamp', 'value'])

avg_inf_speed = duration / X_inf_test.shape[0]
print("INFO: Duration in seconds: ", duration)
print("INFO: Average inference speed of 8 bit model on TPU: ", avg_inf_speed, " seconds per image")
