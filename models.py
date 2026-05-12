import numpy as np
import os
import tensorflow as tf
import pdb

from datetime import datetime, date
from tensorflow.keras import layers, losses, Input
from tensorflow.keras.models import Model

def get_model(
        dimensions,
        input_shape,
        loss_function='mse',
        learning_rate=0.001,
        load_model_path=None
    ):
    model = dncnn(dimensions, input_shape=input_shape)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
    )

    model.build(input_shape=input_shape)
    model.compile(optimizer=optimizer,
                  loss=get_loss(loss_function), 
                  metrics=[]
    )
    
    if load_model_path is not None and load_model_path != '' :
        print('    Loading model weights: {}'.format(load_model_path))
        model.load_weights(load_model_path)
    
    return model

def get_loss(loss_function):
    if loss_function == 'mse':
        return losses.MeanSquaredError()
    elif loss_function == 'mae':
        return losses.MeanAbsoluteError()

def dncnn(
        dimensions,
        input_shape, 
        n_features=64, 
        n_hidden_layers=15, 
        kernel_size=3,
        residual_layer=True
    ):
    
    def conv_layer(dimensions, n_features, activation=None):
        if dimensions == 2:
            return layers.Conv2D(
                    n_features, 
                    (kernel_size, kernel_size), 
                    activation=activation,
                    padding='same', 
                    strides=1
            )
        elif dimensions == 3:
            return layers.Conv3D(
                    n_features, 
                    (kernel_size, kernel_size, kernel_size), 
                    activation=activation,
                    padding='same', 
                    strides=1
            )

    in_layer = Input(shape=input_shape, dtype=tf.float16)

    block = conv_layer(dimensions, n_features)(in_layer)
    block = layers.Activation('relu')(block)

    for i in range(n_hidden_layers):
        block = conv_layer(dimensions, n_features)(block)
        block = layers.BatchNormalization()(block)
        block = layers.Activation('relu')(block)

    output = conv_layer(dimensions, input_shape[-1])(block)
    
    if residual_layer: output = layers.Add()([in_layer, output])

    return Model(inputs=[in_layer], outputs=[output])

def get_training_cb(config_name, patience, save_path):
    # CB: Early stopping
    cb_earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=patience)
    
    # CB: Checkpoint
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Creating new folder for model weights: {}".format(save_path))
    
    save_filename = 'dncnn_ep{epoch:02d}.h5'
    save_filename = os.path.join(save_path, save_filename)
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_filename,
            verbose=1,
            save_best_only=True,
            monitor='val_loss',
            mode='min')

    cb_list = [cb_earlystop, cb_checkpoint]
    
    return cb_list
