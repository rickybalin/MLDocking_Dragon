############# Module Loading ##############
import argparse
import os
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
import horovod.keras as hvd ### importing horovod to use data parallelization in another step

from ST_funcs.clr_callback import *
from ST_funcs.smiles_regress_transformer_funcs import *
from tensorflow.python.client import device_lib
import json

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()

#######HyperParamSetting#############

json_file = 'config.json'
hyper_params = ParamsJson(json_file)

if hyper_params['general']['use_hvd']==True:
    initialize_hvd()

########Create training and validation data##### 
x_train, y_train, x_val, y_val = train_val_data(hyper_params)
#x_train = np.array(x_train).flatten()
#y_train = np.array(y_train).flatten()
print(x_train.shape)

#x_train = tf.data.Dataset.from_tensor_slices(x_train)
#y_train = tf.data.Dataset.from_tensor_slices(y_train)
#print(x_train)
#print(f"x_train shape is {x_train.shape}")
#x_train_batched = np.array_split(x_train, 50)
#y_train_batched = np.array_split(y_train, 50)
#print(x_train_batched)
######## Build model #############

model = ModelArchitecture(hyper_params).call()
model.load_weights('smile_regress.autosave.model.h5')
for layer in model.layers:
    if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
        layer.trainable = False
    #print(f"Layer Name: {layer.name}")
    #print(f"Layer Type: {layer.__class__.__name__}")
    #print(f"Layer Trainable: {layer.trainable}")
    #print(f"Layer Input Shape: {layer.input_shape}")
    #print(f"Layer Output Shape: {layer.output_shape}\n")

model.summary()
####### Set callbacks + train model ##############
#train_dataset = tf.data.Dataset.zip((x_train, y_train)).repeat().batch(64)
#train_dataset = train_dataset.repeat().batch(64)
#for element in train_dataset:
#  print(element)

#print(train_dataset)

#train_and_callbacks = TrainingAndCallbacks(hyper_params)

print(x_train)
print(y_train)
history = model.fit(
    #train_dataset,
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    verbose=1,
    #steps_per_epoch=40,
    validation_data=(x_val,y_val),#validation_data,
    #callbacks=callbacks,
)

# Get the model's weights
weights = model.get_weights()

# Create a dictionary to store the weights
weights_dict = {}

# Iterate over the layers and their respective weights
for layer_idx, layer in enumerate(model.layers):
    for weight_idx, weight in enumerate(layer.get_weights()):
        # Create a key for each weight
        key = f'layer_{layer_idx}_weight_{weight_idx}'
        # Save the weight in the dictionary
        weights_dict[key] = weight

# Print or save the weights dictionary as needed
print(weights_dict)

model = ModelArchitecture(hyper_params).call()
# Assign the weights back to the model
for layer_idx, layer in enumerate(model.layers):
    weights = [weights_dict[f'layer_{layer_idx}_weight_{weight_idx}'] for weight_idx in range(len(layer.get_weights()))]
    layer.set_weights(weights)

#model.load_weights(weights_dict)#'smile_regress.autosave.model.h5')
history = model.fit(
    #train_dataset,
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    verbose=1,
    #steps_per_epoch=40,
    validation_data=(x_val,y_val),#validation_data,
    #callbacks=callbacks,
)

#history = train_and_callbacks.training(
#    model,
#    x_train,
#    y_train,
#    (x_val, y_val),
#    hyper_params
#    )

