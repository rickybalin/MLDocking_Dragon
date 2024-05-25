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
import subprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
#import horovod.keras as hvd ### importing horovod to use data parallelization in another step

from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import *
#from tensorflow.python.client import device_lib

import dragon
from dragon.data.ddict.ddict import DDict

tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()

def training_switch(dd: DDict, candidate_dict: DDict, continue_event, BATCH=128, EPOCH=1000):
    #for i in range(1):
    iter = 0
    
    with open("train_switch.log",'w') as f:
        f.write("Starting inference\n")
    #while continue_event.is_set():
    if True:
        with open("train_switch.log","a") as f:
            f.write(f"Training on iter {iter}\n")
        print(f"Train on iter {iter}",flush=True)
        fine_tune(dd, candidate_dict, BATCH=BATCH, EPOCH=EPOCH)
        
        dd["train_iter"] = iter
        iter += 1
    #print(f"{dd['model_iter']=}")
    #print(f"{dd['model']=}")


def fine_tune(dd: DDict, candidate_dict: DDict, BATCH=8, EPOCH=10):

    ######## Build model #############
    keys = dd.keys()

    if "model" not in keys:
        # On first iteration load pre-trained model
        print(f"Loading pretrained model",flush=True)
        json_file = 'training/config.json'
        hyper_params = ParamsJson(json_file)
        try:
            model = ModelArchitecture(hyper_params).call()
            model.load_weights(f'training/smile_regress.autosave.model.h5')
        except Exception as e:
            with open("train_switch.log","a") as f:
                f.write(f"{e}")
        model_iter = 1
        print(f"Finshed loading pretrained model",flush=True)
        
        with open("train_switch.log","a") as f:
            f.write(f"Finished loading pretrained model\n")
            f.write("\n")
    else:
        model = dd["model"]
        model_iter = dd["model_iter"] + 1


    # Not sure if this is needed, experiment with the print statements in this block
    for layer in model.layers:
        if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
            layer.trainable = False
        #print(f"Layer Name: {layer.name}")
        #print(f"Layer Type: {layer.__class__.__name__}")
        #print(f"Layer Trainable: {layer.trainable}") # check here
        #print(f"Layer Input Shape: {layer.input_shape}")
        #print(f"Layer Output Shape: {layer.output_shape}\n")

    with open("train_switch.log", 'a') as f:
        f.write(f"Create training data\n")
    print(f"Create training data",flush=True)
    ########Create training and validation data##### 
    x_train, y_train = train_val_data(candidate_dict)
    with open("train_switch.log", 'a') as f:
        f.write(f"Finished creating training data\n")
    
    # Only train if there is new data
    if len(x_train) > 0:
        try:
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            train_dataset = train_dataset.batch(BATCH) # Use your desired batch size
            train_dataset = train_dataset.repeat()

        except Exception as e:
            with open("train_switch.log","a") as f:
                f.write(f"{e}")

        steps_per_epoch=int(len(y_train)/BATCH)
        with open("train_switch.log", 'a') as f:
            f.write(f"{BATCH=} {EPOCH=} {steps_per_epoch=}\n")


        try:
            history = model.fit(
                        train_dataset,
                        batch_size=BATCH,
                        epochs=EPOCH,
                        verbose=2,
                        steps_per_epoch=steps_per_epoch,
                        #validation_data=valid_dataset,
                        #callbacks=callbacks,
                    )
        except Exception as e:
            with open("train_switch.log","a") as f:
                f.write(f"{e}")
            

        with open("train_switch.log", 'a') as f:
            f.write(f"Finished fitting model\n")
            f.write(f"history keys {history.history['loss']=}, {history.history['r2']=}\n")

        # model.save("model.keras")
    # Save to dictionary
    try:
        dd["model"] = model
        dd["model_iter"] = model_iter
        if "last_training_docking_iter" in candidate_dict.keys():
            candidate_dict["last_training_docking_iter"] = candidate_dict["last_training_docking_iter"] + 1
        else:
            candidate_dict["last_training_docking_iter"] = 0
    except Exception as e:
        print("writing model to dictionary failed!")
        raise(e)


