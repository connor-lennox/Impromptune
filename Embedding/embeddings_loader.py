import numpy as np
import torch


EMBEDDINGS_ROOT = "Embedding/Embeddings/"
EMBEDDING_2017_240_256 = "2017_240event_256dim.npy"
EMBEDDING_240_256_SMALLWINDOW = "240event_256dim_smallwindow.npy"


def load_embedding(filename):
    return np.load(EMBEDDINGS_ROOT + filename)

