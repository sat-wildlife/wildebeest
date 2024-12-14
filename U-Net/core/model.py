import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras import backend as K
epsilon = 1e-07

from tensorflow.keras import regularizers

import numpy as np

#Tversky loss function, which assigns more weights to false positive/false negative by adjusting alpha and beta parameters
epsilon = 0.000001

weight_set = 0.8

PATCH_SIZE = 336

INPUT_BANDS = [0,1,2]
NUMBER_BANDS=len(INPUT_BANDS)

def tversky(y_true, y_pred, alpha=1-weight_set, beta=weight_set):
    """
    Function to calculate the Tversky loss for imbalanced data
    Args:
      y_true: the segmentation ground_truth
      y_pred: prediction: the logits
      alpha: weight of false positives
      beta: weight of false positives

    Return: the loss
    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    # TP
    true_pos = K.sum(y_true_pos * y_pred_pos)
    # FN
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    # FP
    false_pos = K.sum((1-y_true_pos) * y_pred_pos)
    return 1 - ((true_pos + epsilon)/(true_pos + alpha * false_neg + beta * false_pos + epsilon))

def accuracy(y_true, y_pred, threshold=0.5):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.equal(K.round(y_t), K.round(y_pred))

# K.round() returns the Element-wise rounding to the closest integer
# so the threshold to determine a true positive is here!
def true_positives(y_true, y_pred, threshold=0.5):
    """compute true positive"""
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round(y_true * y_pred)

def false_positives(y_true, y_pred, threshold=0.5):
    """compute false positive"""
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((1 - y_true) * y_pred)

def true_negatives(y_true, y_pred, threshold=0.5):
    """compute true negative"""
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((1 - y_true) * (1 - y_pred))

def false_negatives(y_true, y_pred, threshold=0.5):
    """compute false negative"""
    y_pred = K.round(y_pred +0.5 - threshold)
    return K.round((y_true) * (1 - y_pred))

def recall_m(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    recall = K.sum(tp) / (K.sum(tp) + K.sum(fn)+ epsilon)
    return recall

def precision_m(y_true, y_pred):
    #y_t = y_true[...,0]
    #y_t = y_t[...,np.newaxis]
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    precision = K.sum(tp) / (K.sum(tp) + K.sum(fp)+ epsilon)
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))


#Model builder


def unet(pretrained_weights = None,input_size = (PATCH_SIZE,PATCH_SIZE,NUMBER_BANDS),
         lr = 1.0e-04, drop_out = 0, regularizers = regularizers.l2(0.0001)):
    inputs = Input(input_size)
    # norm0 = BatchNormalization()(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(norm0)    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    norm2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(norm2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    norm3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(norm3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    norm4 = BatchNormalization()(conv4)
    drop4 = Dropout(drop_out)(norm4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(drop_out)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    norm6 = BatchNormalization()(up6)
    merge6 = Concatenate(axis = 3)([drop4, norm6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    norm7 = BatchNormalization()(up7)
    merge7 = Concatenate(axis = 3)([norm3,norm7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    norm8 = BatchNormalization()(up8)
    merge8 = Concatenate(axis = 3)([norm2,norm8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    norm9 = BatchNormalization()(up9)
    merge9 = Concatenate(axis = 3)([norm1,norm9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid', kernel_regularizer= regularizers)(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=lr)
    # OPTIMIZER = Adam(lr= learning_rate, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)
    #OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    #OPTIMIZER = Adam(lr= 1.0e-04, decay= 0.0, beta_1= 0.9, beta_2= 0.999, epsilon= 1.0e-8)
    #OPTIMIZER = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    #OPTIMIZER = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    LOSS = tversky

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[accuracy, precision_m, recall_m, f1_m, 
                                                           true_positives, false_positives, true_negatives, false_negatives])
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model