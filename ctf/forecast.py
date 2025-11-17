# ## Imports

import os
import sys
import time
import torch
import pickle
import argparse
import numpy as np
import pprint as pp
import pandas as pd  # requires: pip install pandas
from pathlib import Path
from datetime import timedelta

# Add top level directory
top_dir = Path(__file__).parent.parent
sys.path.append(str(top_dir))

os.environ['OMP_NUM_THREADS'] = '4'
#os.environ['HF_HOME'] = "/mmfs1/home/alexeyy/storage/.cache/huggingface"
from models.utils import grid_iter
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from data.serialize import SerializerSettings

from ctf4science.data_module import load_validation_dataset, load_dataset, get_prediction_timesteps, get_validation_prediction_timesteps, get_validation_training_timesteps, get_metadata

pickle_dir = top_dir / 'pickles'
ckpt_dir = top_dir / 'checkpoints'
pickle_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

def main(args=None):
    # Start timing from the beginning of main
    main_start_time = time.time()
    
    # ## Model Parameters

    print("> Setting up model parameters")

    llama_hypers = dict(
        temp=1.0,
        alpha=0.99,
        beta=0.3,
        basic=False,
        settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
    )

    model_hypers = {
        'llama-7b': {'model': 'llama-7b', **llama_hypers},
    }

    model_predict_fns = {
        'llama-7b': get_llmtime_predictions_data,
    }

    model_names = list(model_predict_fns.keys())

    model = model_names[0]

    model_hypers[model].update({'dataset_name': args.dataset}) # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10

    # ## Data

    # Pair ids 2, 4: reconstruction
    # Pair ids 1, 3, 5-7: forecast
    # Pair ids 8, 9: burn-in
    pair_id = args.pair_id
    dataset = args.dataset
    validation = args.validation
    recon_ctx = args.recon_ctx # Context length for reconstruction

    print("> Setting up training data")

    md = get_metadata(dataset)

    if validation:
        train_data, val_data, init_data = load_validation_dataset(dataset, pair_id=pair_id)
        forecast_length = get_validation_prediction_timesteps(dataset, pair_id).shape[0]
    else:
        train_data, init_data = load_dataset(dataset, pair_id=pair_id)
        forecast_length = get_prediction_timesteps(dataset, pair_id).shape[0]

    print(f"> Predicting {dataset} for pair {pair_id} with forecast length {forecast_length}")

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        print(f"> Reconstruction task, using {recon_ctx} context length")
        train_mat = train_data[0]
        train_mat = train_mat[0:recon_ctx,:]
        forecast_length = forecast_length - recon_ctx
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        print(f"> Forecasting task, using {forecast_length} forecast length")
        train_mat = train_data[0]
    elif pair_id in [8, 9]:
        # Burn-in
        print(f"> Burn-in matrix of size {init_data.shape[0]}, using {forecast_length} forecast length")
        train_mat = init_data
        forecast_length = forecast_length - init_data.shape[0]
    else:
        raise ValueError(f"Pair id {pair_id} not supported")

    # Set up smaller context and prediction length
    context = min(train_mat.shape[0], 200)
    prediction_length = min(forecast_length, 100)
    test_mat = np.zeros((prediction_length, train_mat.shape[1]))
    print("> Model variables")
    print(f"  Context length: {context}")
    print(f"  Prediction length: {prediction_length}")

    # Check if we have a checkpoint of saved predictions
    ckpt_path = ckpt_dir / f"{args.identifier}.pkl"
    if ckpt_path.exists():
        print(f"> Loading saved predictions from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            pred_ckpt = pickle.load(f)
    else:
        print(f"> No checkpoint found at {ckpt_path}")
        pred_ckpt = {'i': 0, 'train_mat_in': train_mat}

    train_mat_in = pred_ckpt['train_mat_in']

    train_mat_in = train_mat_in.astype(np.float32)
    test_mat = test_mat.astype(np.float32)

    # Forecast each column

    print("> Predicting...")

    forecast_loops = forecast_length // prediction_length + (forecast_length % prediction_length > 0)

    for i in range(pred_ckpt['i'], forecast_loops):
        
        print(f"> ({i+1}/{forecast_loops}) Train Mat Shape:", train_mat_in.shape)

        pred_l = []
        for j in range(train_mat_in.shape[1]):

            # Check if max time has been exceeded
            if args.max_time_hours is not None:
                elapsed_hours = (time.time() - main_start_time) / 3600.0
                if elapsed_hours > args.max_time_hours:
                    print(f"> Maximum time of {args.max_time_hours} hours exceeded ({elapsed_hours:.2f} hours elapsed)")
                    print(f"> Exiting after {i} iterations (out of {forecast_loops})")
                    sys.exit(1)

            start_t = time.time()
            model_hypers[model].update({'dataset_name': args.dataset})
            hypers = list(grid_iter(model_hypers[model]))
            pred_dict = get_autotuned_predictions_data(train_mat_in[-context:,j], test_mat[:,j], hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            median = pred_dict['median'].to_numpy()
            print(f"median shape: {median.shape}")

            pred_l.append(median)
            end_t = time.time()
            str_t = str(timedelta(seconds=end_t-start_t)).split('.')[0]

            print(f"  Predicted dim {j+1} of {train_mat.shape[1]} ({str_t})")
        
        pred_i = np.stack(pred_l, axis=1)
        train_mat_in = np.vstack([train_mat_in, pred_i])
        train_mat_in = train_mat_in.astype(np.float32)

        # Save prediction
        pred_ckpt['i'] = i
        pred_ckpt['train_mat_in'] = train_mat_in
        with open(ckpt_path, "wb") as f:
            pickle.dump(pred_ckpt, f)

    # Save prediction
    raw_pred = train_mat_in[train_mat.shape[0]:train_mat.shape[0]+prediction_length,:]

    print("> Concatenated Shape", raw_pred.shape)

    print("> Creating prediction matrix")

    # Perform pair_id specific operations
    if pair_id in [2, 4]:
        # Reconstruction
        pred = np.vstack([train_mat, raw_pred])
    elif pair_id in [1, 3, 5, 6, 7]:
        # Forecast
        #pred = np.vstack([train_mat, raw_pred])
        pred = raw_pred
    elif pair_id in [8, 9]:
        # Burn-in
        pred = np.vstack([train_mat, raw_pred])
    else:
        raise ValueError(f"Pair id {pair_id} not supported") 

    print("> Predicted Matrix Shape:", pred.shape)
    
    if args.validation:
        print("> Expected Shape: ", val_data.shape)
    else:
        if dataset in ['seismo']:
            print("> Expected Shape: ", md['matrix_shapes'][f'X{pair_id}test.npz'])
        else:
            print("> Expected Shape: ", md['matrix_shapes'][f'X{pair_id}test.mat'])

    # ## Save prediction matrix
    with open(pickle_dir / f"{args.identifier}.pkl", "wb") as f:
        pickle.dump(pred, f)

if __name__ == '__main__':
    # To allow CLIs
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, default=None, required=True, help="Identifier for the run")
    parser.add_argument('--dataset', type=str, default=None, required=True, help="Dataset to run (ODE_Lorenz or PDE_KS)")
    parser.add_argument('--pair_id', type=int, default=1, help="Pair_id to run (1-9)")
    parser.add_argument('--recon_ctx', type=int, default=20, help="Context length for reconstruction")
    parser.add_argument('--validation', type=int, default=0, help="Generate and use validation set")
    parser.add_argument('--device', type=str, default=None, required=True, help="Device to run on")
    parser.add_argument('--max_time_hours', type=float, default=None, help="Maximum time in hours for the forecast loop")
    args = parser.parse_args()

    # Args
    print("> Args:")
    pp.pprint(vars(args), indent=2)

    # Start timing
    start_time = time.time()
    
    main(args)
    
    # End timing and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Convert to HH:MM:SS format
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"> Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
