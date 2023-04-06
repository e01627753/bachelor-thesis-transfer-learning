import numpy as np
import matplotlib.pyplot as plt
import os
import fiftyone

NUM_SAMPLES = 500
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


#code is taken from https://gist.github.com/dipanjanS/4bcf226ae5bb5b1098f13e0dc5527ab7#file-tl_9-py
def plot_model_history(history, epochs=30):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,epochs+1))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, epochs+1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, epochs+1, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")