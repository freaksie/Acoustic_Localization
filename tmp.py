import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import keras
import tensorflow as tf
import tensorflow_addons as tfa
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam, Adamax, SGD
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten , Input
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import mimetypes
import argparse
import imutils
import os
from os import listdir
from os.path import isfile, join
# TF_ENABLE_ONEDNN_OPTS=0

base_path="/home/neel/Acoustic/Acoustics/data"
spectrogram_path=os.path.sep.join([base_path,"images"])
annots_path=os.path.sep.join([base_path,"train.csv"])

base_output="/home/neel/Acoustic/Acoustics/output"
model_path=os.path.sep.join([base_output,"detector.h5"])
plot_path=os.path.sep.join([base_output,"plot.png"])
test_file=os.path.sep.join([base_output,"test.txt"])

from PIL import Image
from numpy import asarray
print("[INFO] loading dataset...")
rows = open(annots_path).read().strip().split("\n")
spectrogram=np.empty(((len(rows)),216,216,2), dtype="float32")
bounding_box_cords=[]
filenames=[]
cnt=0
for row in rows:
    row = row.split(",")
    (filename1,filename2,startX, startY, endX, endY) = row
    imagePath1 = os.path.sep.join([spectrogram_path, filename1])
    imagePath2 = os.path.sep.join([spectrogram_path, filename2])
    image1 = tf.io.read_file(imagePath1)
    image1=tf.image.decode_image(image1,channels=1,dtype=tf.float32) 
    image1=tf.image.resize(image1,[216,216])
    image2 = tf.io.read_file(imagePath2)
    image2=tf.image.decode_image(image2,channels=1,dtype=tf.float32) 
    image2=tf.image.resize(image2,[216,216])
    
    print(image1.shape)
    data=np.concatenate((image1,image2),axis=-1)
    spectrogram[cnt]=data / 255.0
    cnt+=1
    bounding_box_cords.append((startX,startY,endX,endY))
    filenames.append([filename1,filename2])
print("Done.")
#  
