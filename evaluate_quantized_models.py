# -*- coding: utf-8 -*-
'''
Syntax for script execution:
python<optionally the version of your python> evaluate_quantized_models.py -b <number of bits of the model> -m <model name>

EXAMPLE: python3 evaluate_quantized_models.py -b 16 -m tl_model_16f
'''

########################################################################
######### import required libraries
########################################################################
import sys
import os
import argparse
import numpy as np
import time
import fiftyone
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

########################################################################
######### define ArgumentParser, add -b option and parse arguments
########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, help="A numeric option")
parser.add_argument("-m", type=str, help="Ein String als Option")
parser.add_argument("-mode", type=str, help="Ein String als Option")
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
######### run inference and evaluate predictions
########################################################################
#https://learnopencv.com/tensorflow-lite-model-optimization-for-on-device-machine-learning/#Performance-Evaluation-of-TF-Lite-Models-on-Raspberry-Pi
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
prediction = []
output_lst = np.zeros((X_inf_test.shape[0], 2))
for i in range(X_inf_test.shape[0]):
    img = np.expand_dims(X_inf_test[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)

    # run inference
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    duration += end_time-start_time

    # prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = 1 if output[0][0] < 0.5 else 0
    prediction.append(predicted_label)
    
    if (bit == 8): # rescale output data
        output_scale, output_zero_point = output_details[0]['quantization']
        output_lst[i] = output_scale * (output.astype(np.float32) - output_zero_point)

avg_inf_speed = duration / X_inf_test.shape[0]
print("INFO: Duration in seconds:", duration)
print("INFO: Average inference speed of", bit, "bit model:", avg_inf_speed, "seconds per image")

########################################################################
######### evaluate accuracy
########################################################################
prediction = np.array(prediction)
print("First 20 labels of prediction:", prediction[:20])

labels = np.array([
    any(classification['label'] == 'Vehicle registration plate'
        for classification in sample['positive_labels']['classifications'])
    for sample in input_data])

y_input = labels.astype(int)
print("First 20 labels of input_data:", y_input[:20])

accuracy = (prediction == y_input).mean()
print("INFO: Accuracy of model", model_name, "=", accuracy)
