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

#######HyperParamSetting#############

json_file = 'config.json'
hyper_params = ParamsJson(json_file)

if hyper_params['general']['use_hvd']==True:
    initialize_hvd()

########Create training and validation data##### 
x_train, y_train, x_val, y_val = train_val_data(hyper_params)

######## Build model #############

model = ModelArchitecture(hyper_params).call()

####### Set callbacks + train model ##############

train_and_callbacks = TrainingAndCallbacks(hyper_params)

history = train_and_callbacks.training(
    model,
    x_train,
    y_train,
    (x_val, y_val),
    hyper_params
    )

