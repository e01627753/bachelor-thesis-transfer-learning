import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_img(path="data/train/data/00009e5b390986a0.jpg"):

    fig = plt.figure()
    plt.imshow(cv2.imread(path))
    plt.plot()