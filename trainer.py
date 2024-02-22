import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from utils import *
import models
from data import *


def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


seed_tensorflow(2024)

train, valid, test = loadData(144, 300000, 8)
batch_size = 512
x_train, y_train = (train[:, 0:16, 10:].reshape(train.shape[0], -1), train[:, 16:, 0:10]), train[:, 16:, 10:]
tra_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train_batch = tra_dataset.batch(batch_size)

x_valid, y_valid = (valid[:, 0:16, 10:].reshape(valid.shape[0], -1), valid[:, 16:, 0:10]), valid[:, 16:, 10:]
val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

x_test, y_test = (test[:, 0:16, 10:].reshape(test.shape[0], -1), test[:, 16:, 0:10]), test[:, 16:, 10:]
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

model = models.complexNDM(24, 4, 3, mode='sca')


def loss(model, x, y, training):
    y_pred, hidden_states = model(x, training=training)
    loss1 = tf.keras.losses.mae(y, y_pred)
    loss2 = tf.keras.losses.mae(tf.math.abs(hidden_states[0:-1] - hidden_states[1:]),
                                tf.constant(0, shape=tf.shape(hidden_states[0:-1]), dtype=tf.float32))
    loss1, loss2 = tf.math.reduce_mean(loss1), tf.math.reduce_mean(loss2)
    ratio = tf.divide(loss2, loss1)
    ratio = tf.stop_gradient(ratio)

    loss = loss1 + loss2 / ratio
    return loss


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Train
optimizer = tf.keras.optimizers.Nadam(learning_rate=2e-4)
train_loss_results = []
num_epochs = 1000
earlystopping = Early_Stop_Callback(model=model, patience=10, factor=0.5, min_lr=0.00001, repeat=3)
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    print(f"Epoch {epoch}/{num_epochs}")
    progbar = tf.keras.utils.Progbar(len(ds_train_batch))
    for i, (x, y) in enumerate(ds_train_batch):
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        progbar.update(i + 1, [('loss mae', loss_value)])

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    validations, _ = model.predict(x_valid)
    valid_loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(100 * validations - 100 * y_valid)))
    print("Valid Loss RMSE: {:.4f}".format(valid_loss))
    earlystopping(valid_loss)
    if earlystopping.stop_training:
        break

model.load_weights('./checkpoints/best_model')
predictions, _ = model.predict(x_test)
test_loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(100 * predictions - 100 * y_test)))
print("Test Loss RMSE: {:.4f}".format(test_loss))
