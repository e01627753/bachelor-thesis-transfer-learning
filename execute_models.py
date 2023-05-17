########################################################################
######### download, install and import required libraries
########################################################################
import importlib
import subprocess
import sys
import os

libs_to_check = ["argparse", "requests", "numpy", "matplotlib", "time", "fiftyone", "tensorflow"]
libs_to_install = []

for lib in libs_to_check:
    try:
        importlib.import_module(lib)
    except ImportError:
        libs_to_install.append(lib)

if libs_to_install:
    print(f"INFO: Downloading and installing libraries: {', '.join(libs_to_install)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *libs_to_install])
    except Exception as e:
        print(f"ERROR: Could not install following libraries: {e}")
        sys.exit(1)
        
# import libraries
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import fiftyone
import tensorflow as tf
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

########################################################################
######### define ArgumentParser, add -b option and parse arguments
########################################################################5
parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, help="A numeric option")
args = parser.parse_args()

print()
if not args.b:
    print("INFO: Option -b has not been set. 32 bit model will be executed per default.")
    bit = 32
elif args.b != 8 and args.b != 16 and args.b != 32:
    sys.exit("ERROR: You either have to set an option -b of value 6, 16 or 32 or you can take the default 32. No different values are supported!")
else:
    bit = args.b
    
########################################################################
######### download models if not already done
########################################################################
''' TODO
MODELS_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    
#STEPS: check if model is already there -> if not download model
url = "https://raw.githubusercontent.com/user/repo/branch/path/to/file.py"
#https://github.com/e01627753/bachelor-thesis-transfer-learning.git/models
response = requests.get(url)

# Überprüfen Sie den Statuscode der Antwort, um sicherzustellen, dass die Anfrage erfolgreich war
if response.status_code == 200:
    # Speichern Sie die Datei im aktuellen Arbeitsverzeichnis oder einem anderen Pfad
    with open("models/file.py", "wb") as file:
        file.write(response.content)
else:
    # Behandeln Sie den Fehler, wenn die Anfrage fehlschlägt
    print(f"ERROR: could not download {model_name}")

print(f"INFO: loading {bit} bit model...")
'''
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
######### load transfer learning models
########################################################################
if (bit == 8): # 8bit int model
    print(f"INFO: Loading 8 bit model...")
    interpr = tf.lite.Interpreter(model_path="models/tl_model_8int.tflite")
    interpr.allocate_tensors()
    input_details = interpr.get_input_details()
    output_details = interpr.get_output_details()
    
elif (bit == 16): # 16bit float model
    print(f"INFO: Loading 16 bit model...")
    interpr = tf.lite.Interpreter(model_path="models/tl_model_16f.tflite")
    interpr.allocate_tensors()
    input_details = interpr.get_input_details()
    output_details = interpr.get_output_details()
    
else: # 32bit float model
    print(f"INFO: Loading 32 bit model...")
    interpr = tf.lite.Interpreter(model_path="models/tl_model_32f.tflite")
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

        print(f"INFO: Raw inference output scores: {output}")
        print(f"INFO: Output scale: {output_scale}")
        print(f"INFO: Output zero point: {output_zero_point}")
        
        output_lst[i] = output_scale * (output.astype(np.float32) - output_zero_point)

avg_inf_speed = duration / X_inf_test.shape[0]

print(f"INFO: Inference output: {output_lst[0]}")
print(f"INFO: Duration in seconds: {duration}")
print(f"INFO: Average inference speed of {bit} bit model: {avg_inf_speed} seconds per image")

########################################################################
######### for DEBUGGING purpose only
########################################################################
labels = np.array([
    any(classification['label'] == 'Vehicle registration plate'
        for classification in sample['positive_labels']['classifications'])
    for sample in input_data])

y_input = labels.astype(int)
print("First 10 labels of input_data: ", y_input[:20])
#check accuracy -> [0, 1]?!