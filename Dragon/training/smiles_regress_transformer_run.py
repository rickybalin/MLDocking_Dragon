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
#import horovod.keras as hvd ### importing horovod to use data parallelization in another step

from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import *
#from tensorflow.python.client import device_lib

import sys
import os
from time import perf_counter
import datetime

import dragon
from dragon.data.ddict.ddict import DDict

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()

driver_path = os.getenv("DRIVER_PATH")


def fine_tune(dd: DDict, candidate_dict: DDict, BATCH, EPOCH, save_model=True):

    fine_tune_log = "training.log"

    ######## Build model #############
    keys = dd.keys()

    if "model" not in keys:
        # On first iteration load pre-trained model
        json_file = driver_path+'training/config.json'
        hyper_params = ParamsJson(json_file)
        dd["model_hyper_params"] = hyper_params
        try:
            model = ModelArchitecture(hyper_params).call()
            model_path = os.path.join(driver_path,'training/smile_regress.autosave.model.h5')
            model.load_weights(model_path)
        except Exception as e:
            with open(fine_tune_log,"a") as f:
                f.write(f"Failed to load pretrained model from fileystem: {model_path}")
                f.write(f"{e}")
        model_iter = 1
        
        with open(fine_tune_log,"a") as f:
            f.write(f"Finished loading pretrained model\n")
            f.write("\n")
    else:
        try:
            weights_dict = dd["model"]
            model_iter = dd["model_iter"] + 1
            hyper_params = dd["model_hyper_params"]
            model = ModelArchitecture(hyper_params).call()
            # Assign the weights back to the model
            for layer_idx, layer in enumerate(model.layers):
                weights = [weights_dict[f'layer_{layer_idx}_weight_{weight_idx}'] 
                            for weight_idx in range(len(layer.get_weights()))]
                layer.set_weights(weights)

            with open(fine_tune_log,"a") as f:
                f.write(f"Finished loading fine tuned model\n")
                f.write("\n")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            with open(fine_tune_log,'a') as f:
                f.write(f"Failed to load fine tuned model from dictionary\n")
                f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
                f.write(f"{e}\n")
    try:
        for layer in model.layers:
            if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
                layer.trainable = False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        with open(fine_tune_log,'a') as f:
            f.write(f"Failed to set layers to train\n")
            f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
            f.write(f"{e}\n")


    with open(fine_tune_log, 'a') as f:
        f.write(f"Create training data\n")
    ########Create training and validation data##### 
    x_train, y_train, x_val, y_val = train_val_data(candidate_dict)
    with open(fine_tune_log, 'a') as f:
        f.write(f"Finished creating training data\n")
    
    # Only train if there is new data
    if len(x_train) > 0:
        with open(fine_tune_log, 'a') as f:
            f.write(f"{BATCH=} {EPOCH=} {len(x_train)=}\n")
        
        try:
            with open(fine_tune_log, 'a') as sys.stdout:
                history = model.fit(
                            x_train,
                            y_train,
                            batch_size=BATCH,
                            epochs=EPOCH,
                            verbose=2,
                            validation_data=(x_val,y_val),
                            #callbacks=callbacks,
                        )
            weights = model.get_weights()
            weights_dict = {}

            # Iterate over the layers and their respective weights
            for layer_idx, layer in enumerate(model.layers):
                for weight_idx, weight in enumerate(layer.get_weights()):
                    # Create a key for each weight
                    wkey = f'layer_{layer_idx}_weight_{weight_idx}'
                    # Save the weight in the dictionary
                    weights_dict[wkey] = weight

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            with open(fine_tune_log,"a") as f:
                f.write(f"model fit failed\n")
                f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
                f.write(f"{e}")
       
    # Save to dictionary
    try:
        if save_model:
            model_path = "current_model.keras"
            model.save(model_path)
            with open("model_iter",'w') as f:
                f.write(f"{model_iter=} {model_path=}")
       
        dd["model"] = weights_dict
        dd["model_iter"] = model_iter
        print("Saved fine tuned model to dictionary",flush=True)

    except Exception as e:
        print("writing model to dictionary failed!")
        raise(e)
    return history.history



