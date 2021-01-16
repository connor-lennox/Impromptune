import re
import datetime

import torch

from Predictive.Models.predictive_relative_attention_model import PRAm
from Predictive.Models.global_local_models import StackedModel, ParallelModel


pram_regex = r'\d+_pram_k(\d+)_v(\d+)_e(\d+)_r(\d+)_attn(\d+)\.pram'
stacked_regex = r'\d+_stacked_k(\d+)_v(\d+)_e(\d+)_r(\d+)_heads(\d+)_onehot(\d+)_lb(\d+)_lf(\d+).stacked'
parallel_regex = r'\d+_parallel_k(\d+)_v(\d+)_e(\d+)_r(\d+)_heads(\d+)_onehot(\d+)_lb(\d+)_lf(\d+).parallel'


def load_model(model_filename):
    match = re.match(pram_regex, model_filename)
    if match is not None:
        k_d = int(match[1])
        v_d = int(match[2])
        e_d = int(match[3])
        r_d = int(match[4])
        attn_layers = int(match[5])

        net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)
        net.load_state_dict(torch.load(r"TrainedModels\\" + model_filename, map_location=torch.device('cpu')))

        return net

    match = re.match(stacked_regex, model_filename)
    if match is not None:
        k_d = int(match[1])
        v_d = int(match[2])
        e_d = int(match[3])
        r_d = int(match[4])
        heads = int(match[5])
        onehot = bool(int(match[6]))
        lookback = int(match[7])
        lookforward = int(match[8])

        net = StackedModel(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, relative_cutoff=r_d, n_heads=heads,
                           use_onehot_embed=onehot, local_range=(lookback, lookforward))
        net.load_state_dict(torch.load(r"TrainedModels\\" + model_filename, map_location=torch.device('cpu')))

        return net

    match = re.match(parallel_regex, model_filename)
    if match is not None:
        k_d = int(match[1])
        v_d = int(match[2])
        e_d = int(match[3])
        r_d = int(match[4])
        heads = int(match[5])
        onehot = bool(int(match[6]))
        lookback = int(match[7])
        lookforward = int(match[8])

        net = ParallelModel(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, relative_cutoff=r_d, n_heads=heads,
                            use_onehot_embed=onehot, local_range=(lookback, lookforward))
        net.load_state_dict(torch.load(r"TrainedModels\\" + model_filename, map_location=torch.device('cpu')))

        return net


def save_model(model):
    if isinstance(model, PRAm):
        filename = f'{int(datetime.datetime.now().timestamp())}_pram_k{model.key_dim}_v' \
                   f'{model.value_dim}_e{model.embedding_dim}_r{model.relative_cutoff}_attn{model.num_attn_layers}.pram'

    elif isinstance(model, StackedModel):
        filename = f'{int(datetime.datetime.now().timestamp())}_stacked_k{model.key_dim}_v{model.value_dim}_e' \
                   f'{model.embedding_dim}_r{model.relative_cutoff}_heads{model.n_heads}_onehot' \
                   f'{1 if model.use_onehot else 0}_lb{model.local_range[0]}_lf{model.local_range[1]}.stacked'

    elif isinstance(model, ParallelModel):
        filename = f'{int(datetime.datetime.now().timestamp())}_parallel_k{model.key_dim}_v{model.value_dim}_e' \
                   f'{model.embedding_dim}_r{model.relative_cutoff}_heads{model.n_heads}_onehot' \
                   f'{1 if model.use_onehot else 0}_lb{model.local_range[0]}_lf{model.local_range[1]}.parallel'

    else:
        raise TypeError("Invalid model type to save")

    with open(r"TrainedModels" + '\\' + filename, 'wb+') as outfile:
        torch.save(model.state_dict(), outfile)
