import cv2
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import random
from tqdm import tqdm

from sklearn.utils import shuffle
def read_paths(dataset_path, list_path):
    imagepaths = []
    labelpaths = []
    for path in open(list_path):
        md5 = path.split('.')[0]
        imagepaths.append(dataset_path + "images/%s.jpg"%md5)
        labelpaths.append(dataset_path + "spline_labels/%s.json"%md5)
    return imagepaths, labelpaths

val_set_path = '/data/mc_data/MLDC/data/val/'
val_list_path = '/data/mc_data/MLDC/data/val/list.txt'
val_image_paths, val_label_paths = read_paths(val_set_path, val_list_path)

train_set_path = '/data/mc_data/MLDC/data/train_2w/'
train_list_path = 'train_2w_list.txt'
train_image_paths, train_label_paths = read_paths(train_set_path, train_list_path)

def get_label_json(label_paths):
    labels_json = []
    for labelpath in tqdm(label_paths):
        with open(labelpath, 'r') as f:
            data = json.load(f)
            labels_json.append(data)
    return labels_json
print("load label json")
val_labels_json = get_label_json(val_label_paths)
train_labels_json = get_label_json(train_label_paths)

def get_label(labels_json):
    labels = np.zeros((len(labels_json), 224, 224), np.uint8)
    for i in tqdm(xrange(len(labels_json))):
        for j,line in enumerate(labels_json[i].values()[0]):
            for k, point in enumerate(line):
                x, y = int(float(point['x']+.5)), int(float(point['y']+0.5))
                if(x>223):x=223
                if(y>223):
                    y=223
                labels[i][y][x] = 1
                if x < 223: labels[i][y][x+1] = 1
                if x > 0  : labels[i][y][x-1] = 1
                if y < 223: labels[i][y+1][x] = 1
                if y > 0  : labels[i][y-1][x] = 1
    return labels
print("get label")
train_labels = get_label(train_labels_json)
val_labels = get_label(val_labels_json)

def generator(dataset_path, list_path, labels, batch_size, data_shape):
    # read md5 for all image and label
    md5s = []
    for path in open(list_path):
        md5s.append(path.split('.')[0])
    md5s = np.array(md5s)
    
    # generate and preprocess image and label by batch_size
    while True:
        random.seed(batch_size)
        batch_indices = np.random.randint(0, len(labels), batch_size)
        batch_md5s = md5s[batch_indices]
        batch_labels = np.array(labels)[batch_indices]
        batch_labels = batch_labels.reshape(batch_size, -1)
        batch_images = []
        for md5 in batch_md5s:
            img = plt.imread(dataset_path + "images/%s.jpg"%md5)
            img = cv2.resize(img, (data_shape,data_shape), interpolation=cv2.INTER_CUBIC)
            batch_images.append(img)

        yield np.array(batch_images), batch_labels

import pydot
import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Lambda, Cropping2D, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model
from keras.preprocessing import image
from keras import Input

# config gpu used on keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


DATA_SHAPE = 512

def vgg16(input_shape):
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    from keras.applications.vgg16 import VGG16
    path='/home/mc16/download_models/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#     with tf.device('/cpu:0'):
    base_net=VGG16(include_top=False,weights='imagenet',input_shape=input_shape)
    base_net.load_weights(path)
    pre_net = Model(inputs=base_net.input, outputs=base_net.get_layer('block2_conv2').output)
    for layer in pre_net.layers: 
        layer.trainable = False
    return pre_net

def extrate_net(input, pre_net):
#     with tf.device('/cpu:0'):
    features = pre_net(input)
    poll1 = MaxPooling2D(pool_size=(7, 7), strides=(1,1), padding='valid')(features)
    conv1 = Conv2D(128, (9, 9), strides=(1,1), padding='valid')(poll1)
    conv2 = Conv2D(64, (9, 9), strides=(1,1), padding='valid')(conv1)
    conv3 = Conv2D(1, (9, 9), strides=(1,1), padding='valid')(conv2)
    sample_features = AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding='valid')(conv3)
    temp = Flatten()(sample_features)
    extra_net = Model(inputs=input, outputs=temp)
    return extra_net

with tf.device('/cpu:0'):
    input_shape = (DATA_SHAPE,DATA_SHAPE,3)
    X = Input(shape=input_shape)
    vgg_net = vgg16(input_shape)
    vgg_net.summary()
    extra_net = extrate_net(X, vgg_net)
print("model is ok")
extra_net.summary()
extra_net = multi_gpu_model(extra_net,gpus=8)

BATCH_SIZE = 32
train_generator = generator(train_set_path, train_list_path, train_labels, BATCH_SIZE, DATA_SHAPE)
valid_generator = generator(val_set_path, val_list_path, val_labels, BATCH_SIZE, DATA_SHAPE)

print("start train")
LEARNING_RATE = 5e-4
EPOCHS = 10
STEP_PER_EPOCH = int(len(train_image_paths)/BATCH_SIZE)
VALIDATION_STEPS = int(len(val_image_paths)/BATCH_SIZE)
extra_net.compile(loss='mse', optimizer=Adam(LEARNING_RATE))
extra_net.fit_generator(train_generator, 
                    verbose=1,
                    validation_data=valid_generator,
                    steps_per_epoch=STEP_PER_EPOCH,
                    epochs = EPOCHS,
                    validation_steps=10) # VALIDATION_STEPS