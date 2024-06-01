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

#def training_switch(dd: DDict, candidate_dict: DDict, continue_event, BATCH=128, EPOCH=100, checkpoint_interval_min=10):
def training_switch(dd: DDict, 
                    candidate_dict: DDict, 
                    continue_event, 
                    BATCH, EPOCH, 
                    num_top_candidates,
                    checkpoint_interval_min=10):
      
    switch_log = "train_switch.log"
    iter = 0
    
    with open(switch_log,'a') as f:
        f.write(f"{datetime.datetime.now()}: Starting Training\n")
    
    check_time = perf_counter()

    if continue_event is None:
        # if continue_event is None, we are in sequential workflow and don't need to initialize
        last_training_docking_iter = candidate_dict["docking_iter"] - 1
    else:
        last_training_docking_iter = -1
    continue_flag = True
    while continue_flag:
        ckeys = candidate_dict.keys()
        #with open(switch_log,"a") as f:
        #    f.write(f"{ckeys=}\n")
        save_model = False
        # Only retrain if there are fresh simulation resulsts and there are the max number of top candidates
        if "docking_iter" in ckeys:
            docking_iter = candidate_dict["docking_iter"]
        else:
            docking_iter = -1

        num_top_candidates_list = 0
        if "max_sort_iter" in ckeys:
            if candidate_dict["max_sort_iter"] >= "0":
                num_top_candidates_list = len(candidate_dict[candidate_dict["max_sort_iter"]]["inf"])
        with open(switch_log,"a") as f:
            f.write(f"{docking_iter=} {last_training_docking_iter=} {num_top_candidates_list=} {num_top_candidates=}\n")
        if docking_iter > last_training_docking_iter and num_top_candidates_list == num_top_candidates:
            tic = perf_counter()
            with open(switch_log,"a") as f:
                f.write(f"{datetime.datetime.now()}: Training on iter {iter}\n")
            if (check_time - tic)/60. > checkpoint_interval_min:
                save_model = True
                check_time = perf_counter() 
            history = fine_tune(dd, candidate_dict, BATCH=BATCH, EPOCH=EPOCH, save_model=save_model)
            
            dd["train_iter"] = iter
            
            toc = perf_counter()
            with open(switch_log,"a") as f:
                #f.write(f"did not train on iter {iter}")
                f.write(f"{datetime.datetime.now()}: iter {iter}: train time {toc-tic} s\n")
                f.write(f"{datetime.datetime.now()}: iter {iter}: loss={history['loss']}\n")
                f.write(f"{datetime.datetime.now()}: iter {iter}: r2={history['r2']}\n")
            last_training_docking_iter = docking_iter
            iter += 1
        if continue_event == None:
            continue_flag = False
        else:
            continue_flag = continue_event.is_set()
            
        

def fine_tune(dd: DDict, candidate_dict: DDict, BATCH=8, EPOCH=10, save_model=True):

    ######## Build model #############
    keys = dd.keys()

    if "model" not in keys:
        # On first iteration load pre-trained model
        print(f"Loading pretrained model",flush=True)
        json_file = driver_path+'training/config.json'
        hyper_params = ParamsJson(json_file)
        try:
            model = ModelArchitecture(hyper_params).call()
            model.load_weights(driver_path+f'training/smile_regress.autosave.model.h5')
        except Exception as e:
            with open("train_switch.log","a") as f:
                f.write(f"{e}")
        model_iter = 1
        print(f"Finshed loading pretrained model",flush=True)
        
        with open("train_switch.log","a") as f:
            f.write(f"Finished loading pretrained model\n")
            f.write("\n")
    else:
        try:
            model_path = dd["model"]
            model_iter = dd["model_iter"] + 1
            model = keras.saving.load_model(model_path)
            #model = dd["model"]
            #model_iter = dd["model_iter"] + 1
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            with open(f"train_switch.log",'a') as f:
                f.write(f"Failed to load fine tuned model from dictionary\n")
                f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
                f.write(f"{e}\n")


    # Not sure if this is needed, experiment with the print statements in this block
    try:
        for layer in model.layers:
            if layer.name not in ['dropout_3', 'dense_3', 'dropout_4', 'dense_4', 'dropout_5', 'dense_5', 'dropout_6', 'dense_6']:
                layer.trainable = False
            #print(f"Layer Name: {layer.name}")
            #print(f"Layer Type: {layer.__class__.__name__}")
            #print(f"Layer Trainable: {layer.trainable}") # check here
            #print(f"Layer Input Shape: {layer.input_shape}")
            #print(f"Layer Output Shape: {layer.output_shape}\n")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        with open(f"train_switch.log",'a') as f:
            f.write(f"Failed to set layers to train\n")
            f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
            f.write(f"{e}\n")


    with open("train_switch.log", 'a') as f:
        f.write(f"Create training data\n")
    print(f"Create training data",flush=True)
    ########Create training and validation data##### 
    x_train, y_train = train_val_data(candidate_dict)
    
    with open("train_switch.log", 'a') as f:
        f.write(f"Finished creating training data\n")
    
    # Only train if there is new data
    if len(x_train) > 0:
        with open("train_switch.log", 'a') as f:
            f.write(f"{BATCH=} {EPOCH=}\n")
        
        try:
            with open("train_switch.log", 'a') as sys.stdout:
                history = model.fit(
                            x_train,
                            y_train,
                            batch_size=BATCH,
                            epochs=EPOCH,
                            verbose=2,
                            #steps_per_epoch=steps_per_epoch,
                            #validation_data=valid_dataset,
                            #callbacks=callbacks,
                        )
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            with open("train_switch.log","a") as f:
                f.write(f"{exc_type=}, {exc_tb.tb_lineno=}\n")
                f.write(f"{e}")

        # with open("train_switch.log", 'a') as f:
        #     f.write(f"Finished fitting model\n")
        #     f.write(f"history keys {history.history['loss']=}, {history.history['r2']=}\n")

        # model.save("model.keras")
    # Save to dictionary
    try:
        if save_model:
            model_path = "current_model.keras"
            model.save(model_path)
            with open("model_iter",'w') as f:
                f.write(f"{model_iter=} {model_path=}")
        #dd["model"] = model_path
        dd["model"] = model
        dd["model_iter"] = model_iter
        # if "last_training_docking_iter" in candidate_dict.keys():
        #     candidate_dict["last_training_docking_iter"] = candidate_dict["last_training_docking_iter"] + 1
        # else:
        #     candidate_dict["last_training_docking_iter"] = 0
    except Exception as e:
        print("writing model to dictionary failed!")
        raise(e)
    return history.history



