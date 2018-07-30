import numpy as np
import tensorflow as tf
from keras import layers, Input, Model, Sequential, optimizers
from keras.layers import Reshape, Merge, Lambda
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
import keras.backend as K
from keras.engine import Layer
from keras.utils import multi_gpu_model, np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

#checkpoint = ModelCheckpoint('seg.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='max')
#callback_list = [earlystop]
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

GPU_MEMORY_FRACTION = 0.7
DATA_SHAPE = 224
# hyparameters to tune
BATCH_SIZE = 100
LEARNING_RATE = 7e-6
EPOCHS = 120
CLASS_WEIGHT = np.array([1,14])

def segnet(shape=224):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    model = Sequential()
    model.add(Layer(input_shape=(shape , shape ,3)))
    # encoder
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(128, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(256, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(512, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # decoder
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Conv2D(512, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Conv2D(256, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Conv2D(128, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())
    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Conv2D(filter_size, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())
    model.add(Conv2D(2, (1, 1), padding='valid',))
    model.outputHeight = model.output_shape[-2]
    model.outputWidth = model.output_shape[-3] 
    model.add(Activation('softmax'))
    return model

def config_keras_backend(fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = fraction 
    sess = tf.Session(config=config)
    K.set_session(sess)
    
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss
    
    
if __name__ == '__main__':
    print("********** loading labels **********")
    train_labels = np.load('/home/mc16/pre_data/train_label_%s.npy'%DATA_SHAPE)
    val_labels = np.load('/home/mc16/pre_data/val_label_%s.npy'%DATA_SHAPE)
    train_labels = np.concatenate((train_labels,val_labels),axis=0)
    
    print("********** loading images **********")
    train_images = np.load('/home/mc16/pre_data/train_image_%s.npy'%DATA_SHAPE)
    val_images = np.load('/home/mc16/pre_data/val_image_%s.npy'%DATA_SHAPE)
    train_images = np.concatenate((train_images,val_images),axis=0)
    
    print("********** building model **********")
    config_keras_backend(GPU_MEMORY_FRACTION)
    seg = segnet(DATA_SHAPE)
    parallel_seg = multi_gpu_model(seg,gpus=4)
    
    print('********** training... **********')
    loss = weighted_categorical_crossentropy(CLASS_WEIGHT)
    adam = Adam(lr=LEARNING_RATE)
    parallel_seg.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    parallel_seg.fit(x = train_images, 
                         y = train_labels,
                         batch_size = BATCH_SIZE,
                         epochs = EPOCHS,
                         verbose = 1,
                         validation_data = (val_images, val_labels),
                         shuffle = True)
    print('********** saveing mdoel **********')
    seg.save('seg_two.h5')