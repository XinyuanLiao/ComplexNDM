import os
import random
import torch
import numpy as np
from jax import numpy as jnp


def seed_random(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def smoothl1loss(y_true, y_pred):
    abs_loss = jnp.abs(y_true - y_pred)
    square_loss = 0.5 * jnp.square(y_true - y_pred)
    res = jnp.where(abs_loss < 1.0, square_loss, abs_loss - 0.5)
    return jnp.sum(res, axis=-1)
