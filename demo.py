import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import openai
#openai.api_key = os.environ['OPENAI_API_KEY']
#openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    plt.savefig(f'{model_name}.png')
    plt.close()

print(torch.cuda.max_memory_allocated())
print()

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True), 
)

model_hypers = {
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    #'llama-70b': {'model': 'llama-70b', **llama_hypers},
}

model_predict_fns = {
    'llama-7b': get_llmtime_predictions_data,
    #'llama-70b': get_llmtime_predictions_data,
}

model_names = list(model_predict_fns.keys())

datasets = get_datasets()
ds_name = 'AirPassengersDataset'


data = datasets[ds_name]
train, test = data # or change to your own data
out = {}

for model in model_names: # GPT-4 takes a about a minute to run
    model_hypers[model].update({'dataset_name': ds_name}) # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
    out[model] = pred_dict
    print("Model output:")
    print(out[model])
    plot_preds(train, test, pred_dict, model, show_samples=True)
