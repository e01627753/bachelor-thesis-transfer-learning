# -*- coding: utf-8 -*-

########################################################################
######### download, install and import required libraries
########################################################################
import importlib
import subprocess
import sys
import os
'''
# Überprüfe die Python Version
if sys.version_info < (3, 9, 15):
    sys.exit("ERROR: Python version 3.9.15 or higher is required.")

# Aktualisiere pip3 auf die neueste Version
try:
    subprocess.check_call([sys.executable, "-m", "pip3", "install", "--upgrade", "pip"])
except Exception as e:
    print("ERROR: Could not upgrade pip3:", str(e))
    sys.exit(1)
'''
libs_to_install = []
libs_to_check = ["argparse", "requests", "numpy", "time", "fiftyone", "tensorflow"]

for lib in libs_to_check:
    try:
        importlib.import_module(lib)
    except ImportError:
        libs_to_install.append(lib)

if libs_to_install:
    print("INFO: Downloading and installing libraries: " + ', '.join(libs_to_install))
    try:
        subprocess.check_call(["pip3", "install"] + libs_to_install)
    except Exception as e:
        print("ERROR: Could not install following libraries: ", str(e))
        print("INFO: Consider installing the libs manually by executing following command: pip3 install <package-name>")
        sys.exit(1)
        
# import libraries
import argparse
import requests
import numpy as np
import time
import fiftyone
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
######### download test data and reformat data
########################################################################
DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

max_samples=50

# delete previously loaded datasets
for ds_name in fiftyone.list_datasets():
    ds = fiftyone.load_dataset(ds_name)
    ds.delete()

# load datasets and merge them
input_data = fiftyone.zoo.load_zoo_dataset(
    "open-images-v6",
    dataset_name="open-images-v6-inference-test",
    split="test",
    label_types="classifications",
    classes=["Vehicle registration plate"],
    max_samples=max_samples,
    dataset_dir=DATA_DIR,
    only_matching=True
)
dataset_2 = fiftyone.zoo.load_zoo_dataset(
    "open-images-v6",
    dataset_name="dataset_2",
    split="test",
    label_types="classifications",
    classes=["Ball", "Coin", "Fox"],
    max_samples=max_samples,
    dataset_dir=DATA_DIR,
    only_matching=True
)
input_data.add_samples(dataset_2)
dataset_2.delete()

# reformat downloaded data
IMG_DIM = (300, 300) # format should match input details of models
X_input = np.array([img_to_array(load_img(img.filepath, target_size=IMG_DIM)) for img in input_data])
X_input = X_input.astype('float32')
X_input /= 255

print(X_input.shape)
    
########################################################################
######### download models if not already done
########################################################################
MODELS_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    
url = "https://github.com/e01627753/bachelor-thesis-transfer-learning.git/models/" + model_name
print(f"INFO: Downloading {bit} bit model {model_name}...")
response = requests.get(url)

# Überprüfe den Statuscode der Antwort
if response.status_code == 200:
    # Speichern Sie die Datei im aktuellen Arbeitsverzeichnis oder einem anderen Pfad
    with open("models/" + model_name, "wb") as file:
        file.write(response.content)
else:
    # Behandeln Sie den Fehler, wenn die Anfrage fehlschlägt
    print("ERROR: Could not download " + model_name)

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

########################################################################
######### for DEBUGGING purpose only
########################################################################
labels = np.array([
    any(classification['label'] == 'Vehicle registration plate'
        for classification in sample['positive_labels']['classifications'])
    for sample in input_data])

y_input = labels.astype(int)
print("First 20 labels of input_data: ", y_input[:20])
#check accuracy -> [0, 1]?!