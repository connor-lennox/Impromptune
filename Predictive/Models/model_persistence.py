import re
import datetime

import torch

from Predictive.Models.predictive_relative_attention_model import PRAm


model_regex = r'\d+_pram_k(\d+)_v(\d+)_e(\d+)_r(\d+)_attn(\d+)\.pram'


def load_model(model_filename):
    match = re.match(model_regex, model_filename)

    k_d = int(match[1])
    v_d = int(match[2])
    e_d = int(match[3])
    r_d = int(match[4])
    attn_layers = int(match[5])

    net = PRAm(key_dim=k_d, value_dim=v_d, embedding_dim=e_d, num_attn_layers=attn_layers, relative_cutoff=r_d)
    net.load_state_dict(torch.load(r"TrainedModels\\" + model_filename, map_location=torch.device('cpu')))

    return net


def save_model(model):
    filename = f'{int(datetime.datetime.now().timestamp())}_pram_k{model.key_dim}_v' \
               f'{model.value_dim}_e{model.embedding_dim}_r{model.relative_cutoff}_attn{model.num_attn_layers}.pram'

    with open(r"TrainedModels" + '\\' + filename, 'wb+') as outfile:
        torch.save(model.state_dict(), outfile)
