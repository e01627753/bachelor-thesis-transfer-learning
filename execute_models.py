# -*- coding: utf-8 -*-
'''
Syntax for script execution:
python<optionally the version of your python> execute_models.py -b <number of bits of the model> -m <model name>

EXAMPLE: python3 execute_models.py -b 16 -m tl_model_16f
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
import tensorflow as tf

import pandas as pd
from pymeas.device import GPMDevice
from pymeas.output import CsvOutput

########################################################################
######### define ArgumentParser, add -b option and parse arguments
########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, help="A numeric option")
parser.add_argument("-m", type=str, help="Ein String als Option")
args = parser.parse_args()

print()
if not args.b:
    print("INFO: Option -b has not been set. 32 bit model will be executed per default.")
    bit = 32
elif args.b != 8 and args.b != 16 and args.b != 32:
    sys.exit("ERROR: You either have to set an option -b of value 6, 16 or 32 or you can take the default 32. No different values are supported!")
else:
    bit = args.b
    
if not args.m:
    sys.exit("ERROR: You have to set -m as an option to clarify the model name of the model you want to execute! Make sure that this name matches the name of the corresponding model in the github repository /models/<your model name> as well as the bit of option -b.")
else:
    model_name = args.m
    
########################################################################
######### load and reformat test data
########################################################################
IMG_DIM = (300, 300) # format should match input details of models

image_files = os.listdir("data")  # list of image names in /data
num_images = len(image_files)

# initialize empty numpy array
X_input = np.empty((num_images, IMG_DIM[0], IMG_DIM[1], 3), dtype=np.float32)

for i, image_file in enumerate(image_files):
    # path to image
    image_path = os.path.join("data", image_file)
    
    # open and scale image
    image = Image.open(image_path)    
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_DIM)

    # add image to numpy array
    X_input[i] = np.array(image)

print("X_input.shape:", X_input.shape)

########################################################################
######### load quantized tensorflow lite models
########################################################################
# https://www.tensorflow.org/lite/api_docs/java/org/tensorflow/lite/Interpreter#getOutputTensor(int)
# https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
# https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
# https://forum.edgeimpulse.com/t/solved-how-to-run-inference-with-tflite-model-file-from-python-win10/3728

if (bit == 8): # 8bit int model
    print("INFO: Loading 8 bit model:", model_name)
    interpreter = tf.lite.Interpreter(model_path="models/" + model_name + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
elif (bit == 16): # 16bit float model
    print("INFO: Loading 16 bit model:", model_name)
    interpreter = tf.lite.Interpreter(model_path="models/" + model_name + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
else: # 32bit float model
    print("INFO: Loading 32 bit model:", model_name)
    interpreter = tf.lite.Interpreter(model_path="models/" + model_name + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
print("INFO: Input details:")
print(input_details)
print("INFO: Output details:")
print(output_details)


########################################################################
######### connect to power meter
########################################################################
#https://gitlab.com/xyz-user/self-adaptive-moop/-/blob/main/src/measure_energy_consumption.py
power_meter_device = GPMDevice(host="192.168.167.90")
power_meter_device.connect()
measurement_thread = power_meter_device.start_power_capture()

########################################################################
######### run inference and return speed measure
########################################################################
input_type = input_details[0]['dtype']
if (bit == 8): # rescale input data
    input_scale, input_zero_point = input_details[0]['quantization']
    print("Input scale:", input_scale)
    print("Input zero point:", input_zero_point)
    X_inf_test = (X_input / input_scale) + input_zero_point
else:
    X_inf_test = X_input
    
# convert to NumPy array of expected type
X_inf_test = X_inf_test.astype(input_type)

duration = 0.0
output_lst = np.zeros((X_inf_test.shape[0], 2))
# measure power consumption
measurement_thread = power_meter_device.start_power_capture()
for i in range(X_inf_test.shape[0]):
    img = np.expand_dims(X_inf_test[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)

    # run inference
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    duration += end_time-start_time
    print('%.1fms' % ((end_time-start_time) * 1000))

    # prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    if (bit == 8): # rescale output data
        output_scale, output_zero_point = output_details[0]['quantization']
        output_lst[i] = output_scale * (output.astype(np.float32) - output_zero_point)

# stop measuring power consumption
power = power_meter_device.stop_power_capture(measurement_thread)
power_meter_device.disconnect()
data = [{'timestamp': key, 'value': value} for key, value in power.items()]
CsvOutput.save(f"power_measures/{model_name}.csv", field_names=['timestamp', 'value'], data=data)
df = pd.DataFrame(power.items(), columns=['timestamp', 'value'])

avg_inf_speed = duration / X_inf_test.shape[0]
print("INFO: Duration in seconds: ", duration)
print("INFO: Average inference speed of ", bit, " bit model: ", avg_inf_speed, " seconds per image")
