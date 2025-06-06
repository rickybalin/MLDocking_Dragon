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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text

#from .ST_funcs.clr_callback import *
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks, ModelArchitecture, ParamsJson, stratified_sample
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
from time import perf_counter
import csv

import dragon
from dragon.data.ddict.ddict import DDict

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()


def fine_tune( 
                BATCH: int, 
                EPOCH: int, 
                save_model=True):

    tic = perf_counter()
    
    fine_tune_log = "training.log"

    tic_read_model = perf_counter()
    #######HyperParamSetting#############
    driver_path = os.getenv("DRIVER_PATH")
    json_file = os.path.join(driver_path, "training/config.json")
    hyper_params = ParamsJson(json_file)

    ######## Load model #############
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(os.path.join(driver_path,"training/smile_regress.autosave.model.h5"))
    toc_read_model = perf_counter()

    ########Create training and validation data#####
    #with open(fine_tune_log, 'w') as f:
    #    f.write(f"Create training data\n")
    # Read sorted data and split compunds to various processes
    tic_read = perf_counter()
    driver_path = os.getenv("DRIVER_PATH")
    sorted_data_path = driver_path + "/training_data/training_smiles.csv"
    candidates = {"smiles": [], "score": []}
    with open(sorted_data_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                if key == "score":
                    candidates[key].append(float(value))
                else:
                    candidates[key].append(value)
    toc_read = perf_counter()

    x_train, y_train, x_val, y_val = train_val_data(candidates,method="stratified")
    
    ######## Create callbacks #######
    callbacks = assemble_callbacks(hyper_params)

    ######## Train #######
    # Only train if there is new data
    if len(x_train) > 0:
        with open(fine_tune_log, 'a') as f:
            f.write(f"{BATCH=} {EPOCH=} {len(x_train)=}\n")
        
        with open(fine_tune_log, 'a') as sys.stdout:
            history = model.fit(
                        x_train,
                        y_train,
                        batch_size=BATCH,
                        epochs=EPOCH,
                        verbose=2,
                        validation_data=(x_val,y_val),
                        callbacks=callbacks,
                    )
        sys.stdout = sys.__stdout__
        
        # Save to dictionary
        #if save_model:
        #    model_path = "current_model.keras"
        #    model.save(model_path)
        #    with open("model_iter",'w') as f:
        #        f.write(f"{model_path=}")

    tic_write = perf_counter()
    model.save_weights('final_weights.h5')
    toc_write = perf_counter()

    toc = perf_counter()

    io_time = (toc_read-tic_read) + (toc_write-tic_write) + (toc_read_model-tic_read_model)
    print(f"Performed training of {len(candidates['smiles'])} compounds: total={toc-tic}, IO={io_time}",flush=True)

    


