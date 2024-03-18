import os
import random

import numpy as np
import tensorflow as tf


# fixed random seed
def seed_tensorflow(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


@tf.function
def smoothl1loss(y_true, y_pred):
    if y_true.dtype == tf.float64:
        y_true = tf.cast(y_true, tf.float32)
    abs_loss = tf.math.abs(y_true - y_pred)
    square_loss = 0.5 * tf.math.square(y_true - y_pred)
    res = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
    return tf.reduce_sum(res, axis=-1)


class Early_Stop_Callback:
    def __init__(self, model, optimizer, patience=20, factor=0.5, min_lr=0, repeat=3):
        self.model = model
        self.patience = patience
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.repeat = repeat
        self.wait = 0
        self.num_repeats = 0
        self.best_val_loss = float('inf')
        self.stop_training = False

    def __call__(self, current_val_loss):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
            self.model.save_weights('./checkpoints/best_model.keras')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.num_repeats += 1
                if self.num_repeats > self.repeat:
                    print('Reached maximum repeats, stopping training.')
                    self.stop_training = True
                else:
                    self.wait = 0
                    self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                    print(f'Reducing learning rate to {self.optimizer.lr.numpy():.1e} and resetting patience.')
                    self.wait = 0


# save test result to a txt document
def save_experience(filename, data):
    with open(filename, 'a') as f:
        for i in range(len(data)):
            if i < 2:
                f.write(f'{data[i]}' + '\t')
            else:
                f.write(f'{data[i]:.2f}' + '\t')
        f.write('\n')
