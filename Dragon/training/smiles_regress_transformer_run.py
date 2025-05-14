# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import (
#     CSVLogger,
#     EarlyStopping,
#     ModelCheckpoint,
#     ReduceLROnPlateau,
# )
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text

#from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os

import dragon
from dragon.data.ddict.ddict import DDict

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()

class LossCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        # logs contain the loss and any other metrics
        print(f"\nBatch {batch} - Loss: {logs.get('loss'):.4f}")

class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        weights = self.model.trainable_weights
        print(f"\nBatch {batch}:")
        for w in weights:
            print(f"{w.name} mean={tf.reduce_mean(w).numpy():.6f}")


def fine_tune(dd: DDict, 
                candidate_dict: DDict, 
                BATCH: int, 
                EPOCH: int, 
                save_model=True):

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    tf.keras.mixed_precision.set_global_policy('float32')

    fine_tune_log = "training.log"

    ######## Build model #############
        
    model, model_iter, hyper_params = retrieve_model_from_dict(dd)
    model_iter += 1

    for layer in model.layers:
        if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
            layer.trainable = False

    with open(fine_tune_log, 'w') as f:
        f.write(f"Create training data\n")
    ########Create training and validation data##### 
    x_train, y_train, x_val, y_val = train_val_data(candidate_dict)
    with open(fine_tune_log, 'a') as f:
        f.write(f"Finished creating training data\n")
    
    # Only train if there is new data
    if len(x_train) > 0:
        with open(fine_tune_log, 'a') as f:
            f.write(f"{BATCH=} {EPOCH=} {len(x_train)=}\n")
            f.write("\n\nTRAINING DATA\n")
            for n in range(64):
                f.write(f"{x_train[n]}  {y_train[n]}\n")
            f.write("\n\nVALIDATION DATA\n")
            for n in range(64):
                f.write(f"{x_val[n]}  {y_val[n]}\n")
        
        with open(fine_tune_log, 'a') as sys.stdout:
            output = model.predict(x_train[:64], batch_size=64, verbose=0).flatten()
            print('predictions: ',output)
            print('truth: ',y_train[:64].flatten())
            print('mse: ',np.mean(np.square(output - y_train[:64].flatten())))
            history = model.fit(
                        x_train[:64],
                        y_train[:64],
                        batch_size=BATCH,
                        epochs=EPOCH,
                        verbose=2,
                        validation_data=(x_val[:64],y_val[:64]),
                        shuffle=True,
                        callbacks=[DebugCallback(),LossCallback()],
                    )
            print("model fitting complete",flush=True)
        sys.stdout = sys.__stdout__
        print("model fitting complete",flush=True)
        
        
        # Save to dictionary
        if save_model:
            model_path = "current_model.keras"
            model.save(model_path)
            with open("model_iter",'w') as f:
                f.write(f"{model_iter=} {model_path=}")

        save_model_weights(dd, model, model_iter)
        print("Saved fine tuned model to dictionary",flush=True)
    


