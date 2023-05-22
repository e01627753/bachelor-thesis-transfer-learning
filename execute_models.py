# -*- coding: utf-8 -*-

########################################################################
######### download, install and import required libraries
########################################################################
import importlib
import subprocess
import sys
import os

libs_to_install = []
libs_to_check = ["argparse", "requests", "numpy", "time", "PIL", "tensorflow"]

for lib in libs_to_check:
    try:
        importlib.import_module(lib)
    except ImportError:
        libs_to_install.append(lib)

if libs_to_install:
    print("INFO: Downloading and installing libraries: " + ', '.join(libs_to_install))
    try:
        subprocess.check_call(["pip", "install"] + libs_to_install)
    except Exception as e:
        print("ERROR: Could not install following libraries: ", str(e))
        print("INFO: Consider installing the libs manually by executing following command: pip3 install <package-name>")
        sys.exit(1)
        
# import libraries
import argparse
import requests
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

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
######### load transfer learning models
########################################################################
if (bit == 8): # 8bit int model
    print("INFO: Loading 8 bit model:", model_name)
    interpr = tf.lite.Interpreter(model_path="models/" + model_name)
    interpr.allocate_tensors()
    input_details = interpr.get_input_details()
    output_details = interpr.get_output_details()
    
elif (bit == 16): # 16bit float model
    print("INFO: Loading 16 bit model:", model_name)
    interpr = tf.lite.Interpreter(model_path="models/" + model_name)
    interpr.allocate_tensors()
    input_details = interpr.get_input_details()
    output_details = interpr.get_output_details()
    
else: # 32bit float model
    print("INFO: Loading 32 bit model:", model_name)
    interpr = tf.lite.Interpreter(model_path="models/" + model_name)
    interpr.allocate_tensors()
    input_details = interpr.get_input_details()
    output_details = interpr.get_output_details()
    
print("INFO: Input details:")
print(input_details)
print("INFO: Output details:")
print(output_details)

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
for i in range(X_inf_test.shape[0]):
    img = np.expand_dims(X_inf_test[i], axis=0)
    interpr.set_tensor(input_details[0]['index'], img)

    # run inference
    start_time = time.time()
    interpr.invoke()
    end_time = time.time()
    duration += end_time-start_time

    # output_details[0]['index'] = the index which provides the input
    output = interpr.get_tensor(output_details[0]['index'])
    
    if (bit == 8): # rescale output data
        output_scale, output_zero_point = output_details[0]['quantization']

        print("INFO: Raw inference output scores: ", output)
        print("INFO: Output scale: ", output_scale)
        print("INFO: Output zero point: ", output_zero_point)
        
        output_lst[i] = output_scale * (output.astype(np.float32) - output_zero_point)

avg_inf_speed = duration / X_inf_test.shape[0]

print("INFO: Inference output: ", output_lst[0])
print("INFO: Duration in seconds: ", duration)
print("INFO: Average inference speed of ", bit, " bit model: ", avg_inf_speed, " seconds per image")