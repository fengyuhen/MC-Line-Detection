import numpy as np
import tensorflow as tf
from keras import layers, Input, Model, Sequential, optimizers
from keras.layers import Reshape, Merge, Lambda, Add, Subtract, dot
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer
from keras.utils import multi_gpu_model, np_utils
from keras.models import load_model

GPU_MEMORY_FRACTION = 0.5
DATA_SHAPE = 224
# hyparameters to tune
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
EPOCHS = 20
CLASS_WEIGHT = np.array([1,15])

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
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(512, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(256, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(128, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(pool_size,pool_size)))
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(2, (1, 1), padding='valid',))
    model.outputHeight = model.output_shape[-2]
    model.outputWidth = model.output_shape[-3] 
    model.add(Activation('softmax'))
    return model

def mycrf(unary, image): 
    q0 = unary
    
    f1 = Conv2D(2, (3, 3), padding='same')(image)
    f1 =  BatchNormalization()(f1)
    m1 = Lambda(lambda x: K.tf.multiply(x[0], x[1]))([f1, q0])
    r1 = Conv2D(2, (1, 1), padding='same')(m1) 
    r1 =  BatchNormalization()(r1)
    p1 = Conv2D(2, (1, 1), padding='same')(r1)
    p1 =  BatchNormalization()(p1)
    q1 = Subtract()([unary, p1])
    q1 = Activation('softmax')(q1)
    
    f2 = Conv2D(2, (3, 3), padding='same')(image)
    f2 =  BatchNormalization()(f2)
    m2 = Lambda(lambda x: K.tf.multiply(x[0], x[1]))([f2, q1])
    r2 = Conv2D(2, (1, 1), padding='same')(m2)
    r2 =  BatchNormalization()(r2)
    p2 = Conv2D(2, (1, 1), padding='same')(r2)
    p2 =  BatchNormalization()(p2)
    q2 = Subtract()([unary, p2])
    q2 = Activation('softmax')(q2)
    
    f3 = Conv2D(2, (3, 3), padding='same')(image)
    f3 =  BatchNormalization()(f3)
    m3 = Lambda(lambda x: K.tf.multiply(x[0], x[1]))([f3, q2])
    r3 = Conv2D(2, (1, 1), padding='same')(m3)
    r3 =  BatchNormalization()(r3)
    p3 = Conv2D(2, (1, 1), padding='same')(r3)
    p3 =  BatchNormalization()(p3)
    q3 = Subtract()([unary, p3])
    q3 = Activation('softmax')(q3)
    
    f4 = Conv2D(2, (3, 3), padding='same')(image)
    f4 =  BatchNormalization()(f4)
    m4 = Lambda(lambda x: K.tf.multiply(x[0], x[1]))([f4, q3])
    r4 = Conv2D(2, (1, 1), padding='same')(m4)
    r4 =  BatchNormalization()(r4)
    p4 = Conv2D(2, (1, 1), padding='same')(r4)
    p4 =  BatchNormalization()(p4)
    q4 = Subtract()([unary, p4])
    q4 = Activation('softmax')(q4)
    
    f5 = Conv2D(2, (3, 3), padding='same')(image)
    f5 =  BatchNormalization()(f5)
    m5 = Lambda(lambda x: K.tf.multiply(x[0], x[1]))([f5, q4])
    r5 = Conv2D(2, (1, 1), padding='same')(m5)
    r5 =  BatchNormalization()(r5)
    p5 = Conv2D(2, (1, 1), padding='same')(r5)
    p5 =  BatchNormalization()(p5)
    q5 = Subtract()([unary, p5])
    q5 = Activation('softmax')(q5)
    
    return q5

def mysegcrf(shape):
    input_img = Input(shape=(shape, shape, 3))
    input_nor = Lambda(lambda x: x/127.5 - 1.)(input_img)
    seg = segnet(DATA_SHAPE)
    # seg = load_model('seg0608.h5')
    for layer in seg.layers:
        layer.trainable = False
    unary = seg(input_nor)
    q_value = mycrf(unary, input_nor)
    model = Model(inputs=input_img, outputs=q_value)
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
    
    print("********** loading images **********")
    train_images = np.load('/home/mc16/pre_data/train_image_%s.npy'%DATA_SHAPE)
    val_images = np.load('/home/mc16/pre_data/val_image_%s.npy'%DATA_SHAPE)
    
    print("********** building model **********")
    config_keras_backend(GPU_MEMORY_FRACTION)
    segcrf = mysegcrf(DATA_SHAPE)
    parallel_segcrf = multi_gpu_model(segcrf,gpus=8)
    
    print('********** training... **********')
    loss = weighted_categorical_crossentropy(CLASS_WEIGHT)
    adam = Adam(lr=LEARNING_RATE)
    parallel_segcrf.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    parallel_segcrf.fit(x = train_images, 
                         y = train_labels,
                         batch_size = BATCH_SIZE,
                         epochs = EPOCHS,
                         verbose = 1,
                         validation_data = (val_images, val_labels),
                         shuffle = True)
    print('********** saveing mdoel **********')
    segcrf.save('segcrf.h5')