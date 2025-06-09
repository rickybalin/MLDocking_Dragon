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
from .ST_funcs.smiles_regress_transformer_funcs import train_val_data, assemble_callbacks
from data_loader.model_loader import retrieve_model_from_dict, save_model_weights
import sys
import os
from time import perf_counter

import dragon
from dragon.data.ddict.ddict import DDict

#tf.config.run_functions_eagerly(True)
#tf.enable_eager_execution()


def fine_tune(model_dd: DDict, 
                sim_dd: DDict, 
                BATCH: int, 
                EPOCH: int, 
                save_model=False,
                debug=False):

    fine_tune_log = "training.log"
    tic_start = perf_counter()

    ######## Build model #############
    tic = perf_counter()
    model, hyper_params = retrieve_model_from_dict(model_dd,fine_tune=True)
    ddict_time = perf_counter() - tic

    ########Create training and validation data#####
    x_train, y_train, x_val, y_val, time = train_val_data(sim_dd,method="stratified")
    ddict_time += time
    with open(fine_tune_log, 'w') as f:
        f.write(f"{BATCH=} {EPOCH=} {len(x_train)=} {len(x_val)=}\n")
    
    ######## Create callbacks #######
    callbacks = assemble_callbacks(hyper_params)

    ######## Train #######
    # Only train if there is new data
    if len(x_train) > 0:        
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
        
        # Save to file
        if save_model:
            model_path = "current_model.keras"
            model.save(model_path)
            with open("model_iter",'w') as f:
                f.write(f"{model_path=}")

        # Save to DDict
        tic = perf_counter()
        save_model_weights(model_dd, model)
        ddict_time += perf_counter() - tic
        toc_end = perf_counter()

        print(f"Performed training: total={toc_end-tic_start}, IO={ddict_time}",flush=True)
    


