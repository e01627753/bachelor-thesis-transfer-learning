import numpy as np
import os

import fiftyone

import tensorflow as tf
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

NUM_SAMPLES = 500
IMG_DIM = (224, 224)
DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
def load_fiftyone_dataset(datasetType):
    # set max_samples
    if(datasetType == "train"):
        max_samples=NUM_SAMPLES
    else:
        max_samples=int(NUM_SAMPLES/10)
        
    # delete previously loaded datasets
    if(len(fiftyone.list_datasets()) > 2):
        for ds_name in fiftyone.list_datasets():
            ds = fiftyone.load_dataset(ds_name)
            ds.delete()
    
    # load datasets and merge them
    dataset_1 = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        dataset_name="open-images-v6-" + datasetType,
        split=datasetType,
        label_types="classifications",
        classes=["Vehicle registration plate"],
        max_samples=max_samples,
        dataset_dir=DATA_DIR,
        only_matching=True
    )
    dataset_2 = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        dataset_name="dataset_2",
        split=datasetType,
        label_types="classifications",
        classes=["Dog", "Cat", "Apple"],
        max_samples=max_samples,
        dataset_dir=DATA_DIR,
        only_matching=True
    )
    dataset_1.add_samples(dataset_2)
    dataset_2.delete()
    
    return dataset_1

# load images of a fiftyone dataset view into a np.array
def load_datasetImages_into_npArray(datasetView):
    samples = np.array([img_to_array(load_img(img.filepath, target_size=IMG_DIM)) for img in datasetView])
    samples = samples.astype('float32')
    samples /= 255
    return samples

# load labels of a fiftyone dataset view into a np.array
def load_datasetLabels_into_npArray(datasetView):
    labels = np.array([
        any(classification['label'] == 'Vehicle registration plate'
            for classification in sample['positive_labels']['classifications'])
        for sample in datasetView])
    return labels.astype(int)

### TensorFlow conversions
#TF Dataset: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#attributes_1
#numpy as TFDataset:
#https://www.tensorflow.org/guide/data#reading_input_data
#https://www.tensorflow.org/tutorials/load_data/numpy
def load_imageTensor(path):
    image_raw = tf.io.read_file(path)
    image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    return image_tensor

def load_datasetImages_and_Labels_into_DataLoader(datasetView, labels):
    #get all filepaths
    all_image_paths = []
    for image in datasetView:
        all_image_paths.append(image.filepath)
        
    #set label names = index to labels (0 = Others, 1 = Vehicle registration plate)
    label_names = ['Others', 'Vehicle registration plate']
    
    #load tensorflow datasets
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_imageTensor)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return DataLoader(image_label_ds, len(all_image_paths), label_names)
