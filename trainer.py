import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import *
from models import *
from data import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--prediction_length', type=int, default=16)
    parser.add_argument('--estimation_length', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=200000)
    parser.add_argument('--down_rate', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden state space size')
    parser.add_argument('--output_size', type=int, default=4)
    parser.add_argument('--layer_num', type=int, default=3, help='number of hidden layers of f_0 and f_u')
    parser.add_argument('--phase', type=float, default=np.pi / 10, help='phase range of eigenvalues')
    parser.add_argument('--scan', type=bool, default=True, help='parallel or serial')
    args = parser.parse_args()
    return args


def Loss(model, x, y, training):
    y_pred, hidden_states = model(x, training=training)
    # Inference Loss
    loss_inf = smoothl1loss(y, y_pred)
    # Smooth Loss
    loss_smth = smoothl1loss(tf.math.abs(hidden_states[0:-1] - hidden_states[1:]),
                             tf.constant(0, shape=tf.shape(hidden_states[0:-1]), dtype=tf.float32))
    loss_inf, loss_smth = tf.math.reduce_mean(loss_inf), tf.math.reduce_mean(loss_smth)

    # The factor to uniform the scale between loss_inf and loss_smth
    ratio = tf.divide(loss_smth, loss_inf)
    ratio = tf.stop_gradient(ratio)
    
    # coefficient Q is 0.1
    loss = loss_inf + loss_smth / (10 * ratio)
    return loss, loss_inf, loss_smth


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value, loss_inf, loss_smth = Loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), loss_inf, loss_smth


# Train
def trainer(arguments):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # data load
    train, valid, test = loadData((arguments.prediction_length,
                                  arguments.estimation_length),
                                  arguments.num_samples,
                                  arguments.down_rate)
    print("train shape: ", train.shape)
    print("valid shape: ", valid.shape)
    print("test shape: ", test.shape)

    x_train, y_train = (train[:, 0:arguments.prediction_length, 10:].reshape(train.shape[0], -1),
                        train[:, arguments.prediction_length:, 0:10]), train[:, arguments.prediction_length:, 10:]

    # buffer_size is a little bigger than the amount of the training data to upset the data within per batch
    batch_size, buffer_size = 1024, int(train.shape[0] * 1.2)
    
    tra_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size)
    ds_train_batch = tra_dataset.batch(batch_size)

    x_valid, y_valid = (valid[:, 0:arguments.prediction_length, 10:].reshape(valid.shape[0], -1),
                        valid[:, arguments.prediction_length:, 0:10]), valid[:, arguments.prediction_length:, 10:]

    x_test, y_test = (test[:, 0:arguments.prediction_length, 10:].reshape(test.shape[0], -1),
                      test[:, arguments.prediction_length:, 0:10]), test[:, arguments.prediction_length:, 10:]

    # model setup and build
    model = complexNDM(hidden_size=arguments.hidden_size,
                       output_size=arguments.output_size,
                       layer_num=arguments.layer_num,
                       scan=arguments.scan,
                       phase=arguments.phase)
    model.build(input_shape=[(10, 64), (10, 128, 10)])
    model.summary()

    # train
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    train_loss_results = []
    num_epochs = 10000

    # see utils.py for the details
    earlystopping = Early_Stop_Callback(model=model, 
                                        optimizer=optimizer, 
                                        patience=20, 
                                        factor=0.5, 
                                        min_lr=0.00001,
                                        repeat=3)

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        print(f"Epoch {epoch}/{num_epochs}")
        progbar = tf.keras.utils.Progbar(len(ds_train_batch))
        for i, (x, y) in enumerate(ds_train_batch):
            loss_value, grads, loss_inf, loss_smth = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_inf)
            progbar.update(i + 1, [('Loss_inf', loss_inf),
                                   ('Loss_smth', loss_smth)])
        train_loss_results.append(epoch_loss_avg.result())

        # valid and early stop
        validations, _ = model(x_valid, training=False)
        valid_loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(100 * validations - 100 * y_valid)))
        print("Valid Loss RMSE: {:.4f}".format(valid_loss)+'\n')

        earlystopping(valid_loss)
        if earlystopping.stop_training:
            break

    # test
    # load the best model parameter
    model.load_weights('./checkpoints/best_model.keras')
    
    predictions, _ = model(x_test, training=False)
    
    test_loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(100 * predictions - 100 * y_test)))
    # maximum of the error, i.e., infinite norm
    l_max = tf.math.reduce_max(tf.math.abs(100 * predictions - 100 * y_test))
    print("Test Loss RMSE: {:.4f}".format(test_loss))
    print("Test Loss l_max: {:.4f}".format(l_max))


def main():
    args = parse_arguments()
    seed_tensorflow(args.seed)
    trainer(args)


if __name__ == '__main__':
    main()
