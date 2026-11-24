"""Batched (multi-channel) LLMTime sampling for Llama backends.

Method-preserving speedup of the per-channel loop in ctf/forecast.py:
for every channel the scaler, serialization, truncation, token budget,
sampling settings (temp, top_p=0.9, digit/comma-only vocabulary), number
of samples and median aggregation are exactly those of
models.llmtime.get_llmtime_predictions_data + models.llama.llama_completion_fn.
The only differences are that several channels (each with all of its
samples) share one left-padded `generate` call with an attention mask, and
each row is cut back to that channel's own `max_new_tokens` budget.
"""
import numpy as np
import pandas as pd
import torch

from models.llmtime import get_scaler, truncate_input, handle_prediction, STEP_MULTIPLIER
from models.llama import get_model_and_tokenizer
from data.serialize import serialize_arr, deserialize_str

_BAD_WORDS_CACHE = {}


def _bad_words(tokenizer, settings):
    key = settings.time_sep
    if key not in _BAD_WORDS_CACHE:
        good = {tokenizer.convert_tokens_to_ids(t) for t in list("0123456789" + settings.time_sep)}
        _BAD_WORDS_CACHE[key] = [[i] for i in range(len(tokenizer)) if i not in good]
    return _BAD_WORDS_CACHE[key]


def _prepare(arr, steps, settings, alpha, beta, basic, model_key, tokenizer):
    scaler = get_scaler(arr, alpha=alpha, beta=beta, basic=basic)
    input_str = serialize_arr(scaler.transform(arr), settings)
    _, input_str = truncate_input(arr, input_str, settings, model_key, steps)
    # identical to llama_completion_fn (steps passed there is steps*STEP_MULTIPLIER)
    n_ids = len(tokenizer(input_str)['input_ids'])
    avg_tokens_per_step = n_ids / len(input_str.split(settings.time_sep))
    max_tokens = int(avg_tokens_per_step * steps * STEP_MULTIPLIER)
    return scaler, input_str, max_tokens


def batched_medians(train_cols, steps, settings, num_samples=10, temp=1.0, top_p=0.9,
                    alpha=0.99, beta=0.3, basic=False, model='7b', model_key='llama-7b',
                    max_seqs=40):
    """Return array (steps, n_channels) of per-channel sample medians.

    train_cols: list of 1-D float arrays (context of each channel).
    max_seqs: maximum number of sequences (channels*num_samples) per generate call.
    """
    llm, tokenizer = get_model_and_tokenizer(model, cache_model=True)
    bad_words = _bad_words(tokenizer, settings)
    prepped = [_prepare(np.asarray(a), steps, settings, alpha, beta, basic, model_key, tokenizer)
               for a in train_cols]
    ch_per_call = max(1, max_seqs // num_samples)
    medians = [None] * len(train_cols)
    idx = list(range(len(train_cols)))
    for s in range(0, len(idx), ch_per_call):
        group = idx[s:s + ch_per_call]
        strs = [prepped[j][1] for j in group]
        budgets = [prepped[j][2] for j in group]
        batch = tokenizer(strs, return_tensors='pt', padding=True, padding_side='left')
        batch = {k: v.cuda() for k, v in batch.items()}
        n_in = batch['input_ids'].shape[1]
        with torch.no_grad():
            out = llm.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max(budgets),
                temperature=temp,
                top_p=top_p,
                bad_words_ids=bad_words,
                renormalize_logits=True,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, n_in:].cpu()
        for g, j in enumerate(group):
            scaler = prepped[j][0]
            rows = gen[g * num_samples:(g + 1) * num_samples, :budgets[g]]
            comps = tokenizer.batch_decode(rows, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            preds = []
            for c in comps:
                p = handle_prediction(deserialize_str(c, settings, ignore_last=False, steps=steps),
                                      expected_length=steps, strict=False)
                preds.append(scaler.inv_transform(p) if p is not None else None)
            medians[j] = pd.DataFrame(preds, columns=pd.RangeIndex(steps)).median(axis=0).to_numpy()
    return np.stack(medians, axis=1)
