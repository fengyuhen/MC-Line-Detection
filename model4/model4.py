import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.models import *
from keras.layers import *
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing import image
from keras import Input
import keras.backend as KB
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2
from keras.engine import Layer

GPU_MEMORY_FRACTION = 0.5
# hyparameters
DATA_SHAPE = 224
BATCH_SIZE = 256
LEARNING_RATE = 5e-5
EPOCHS = 20
CLASS_WEIGHT = np.array([1,13]) # positive:negative

# config the backend for keras, especially the usage of GPU
def config_keras_backend(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    sess = tf.Session(config=config)
    KB.set_session(sess)
    
###### fcn ######
# extrate encode features
def get_encode_features(input_img):
    path='/home/mc16/download_models/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    vgg16 = VGG16(include_top=False,weights='imagenet', input_tensor=input_img)
    vgg16.load_weights(path)
    for layer in vgg16.layers:
        layer.trainable = False
    pool1 = vgg16.get_layer('block1_pool').output
    pool2 = vgg16.get_layer('block2_pool').output
    pool3 = vgg16.get_layer('block3_pool').output
    pool4 = vgg16.get_layer('block4_pool').output
    pool5 = vgg16.get_layer('block5_pool').output
    
    return pool1, pool2, pool3, pool4, pool5

def get_decode_features(pool1, pool2, pool3, pool4, pool5):
    fc1 = Conv2D(filters=2, kernel_size=(1,1), strides=(1,1))(pool5)
    deconv1 = Deconv2D(pool4.shape.as_list()[-1], (4,4), strides=(2,2), padding='SAME')(pool5)
    fc2 = layers.add([deconv1, pool4])
    deconv2 = Deconv2D(pool3.shape.as_list()[-1], (4,4), strides=(2,2), padding='SAME')(fc2)
    fc3 = layers.add([deconv2, pool3])
    deconv3 = Deconv2D(pool2.shape.as_list()[-1], (4,4), strides=(2,2), padding='SAME')(fc3)
    fc4 = layers.add([deconv3, pool2])
    deconv4 = Deconv2D(pool1.shape.as_list()[-1], (4,4), strides=(2,2), padding='SAME')(fc4)
    fc5 = layers.add([deconv4, pool1])
    fc6 = Deconv2D(2, (4,4), strides=(2,2), padding='SAME')(fc5)
    return fc6

def fcn(shape):
    input_img = Input(shape=(shape, shape, 3))
    pool1, pool2, pool3, pool4, pool5 = get_encode_features(input_img)
    decode_features = get_decode_features(pool1, pool2, pool3, pool4, pool5)
    softmax_logits = Activation('softmax')(decode_features)
    model = Model(inputs=input_img, outputs=softmax_logits) 
    return model

def weighted_categorical_crossentropy(weights):
    weights = KB.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= KB.sum(y_pred, axis=-1, keepdims=True)
        y_pred = KB.clip(y_pred, KB.epsilon(), 1 - KB.epsilon())
        loss = y_true * KB.log(y_pred) * weights
        loss = -KB.sum(loss, -1)
        return loss
    return loss

if __name__ == '__main__':
    print("********** load train label **********")
    train_labels = np.load('/home/mc16/pre_data/train_label_%s.npy'%DATA_SHAPE)
    print("********** load val label **********")
    val_labels = np.load('/home/mc16/pre_data/val_label_%s.npy'%DATA_SHAPE)
    
    print("********** load train image **********")
    train_images = np.load('/home/mc16/pre_data/train_image_%s.npy'%DATA_SHAPE)
    print("********** load val image **********")
    val_images = np.load('/home/mc16/pre_data/val_image_%s.npy'%DATA_SHAPE)
    
    print("********** building the fcn mdoel **********")
    config_keras_backend(GPU_MEMORY_FRACTION)
    fcn = fcn(DATA_SHAPE)
    parallel_fcn = multi_gpu_model(fcn,gpus=8) 
    
    print('********** training the fcn mdoel **********')
    loss = weighted_categorical_crossentropy(CLASS_WEIGHT)
    adam = optimizers.Adam(lr=LEARNING_RATE)
    parallel_fcn.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    parallel_fcn.fit(x = train_images, 
                     y = train_labels,
                     batch_size = BATCH_SIZE,
                     epochs = EPOCHS,
                     verbose = 1,
                     validation_data = (val_images, val_labels),
                     shuffle = True)
    
    print('********** save the fcn mdoel **********')
    model_json = fcn.to_json()
    open('model4_structure.json','w').write(model_json)
    fcn.save_weights('model4_weights.h5')