import argparse
import flax
from torch.utils.data import DataLoader, TensorDataset
from flax.training import train_state
import optax
from tqdm import tqdm

from utils import *
from models import *
from data import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--prediction_length', type=int, default=8)
    parser.add_argument('--estimation_length', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=200000)
    parser.add_argument('--down_rate', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=16, help='hidden state space size')
    parser.add_argument('--output_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--layer_num', type=int, default=2, help='number of hidden layers of f_0 and f_u')
    parser.add_argument('--phase', type=float, default=jnp.pi / 10, help='phase range of eigenvalues')
    parser.add_argument('--r_max', type=float, default=1.0)
    parser.add_argument('--r_min', type=float, default=0.9)
    parser.add_argument('--is_PILF', action="store_true", help='whether use the loss_smth')
    parser.add_argument('--scan', action="store_true", help='parallel or serial')
    args = parser.parse_args()
    return args


def trainer(arguments):
    train, valid, test = loadData((arguments.prediction_length,
                                   arguments.estimation_length),
                                  arguments.num_samples,
                                  arguments.down_rate)
    print("train shape: ", train.shape)
    print("valid shape: ", valid.shape)
    print("test shape: ", test.shape)

    train = torch.tensor(train, dtype=torch.float32)

    batch_size = 1024

    dataset = TensorDataset(
        train[:, 0:arguments.prediction_length, 10:].reshape(train.shape[0], -1),
        train[:, arguments.prediction_length:, 0:10],
        train[:, arguments.prediction_length:, 10:]
    )
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_valid, y_valid = (valid[:, 0:arguments.prediction_length, 10:].reshape(valid.shape[0], -1),
                        valid[:, arguments.prediction_length:, 0:10]), valid[:, arguments.prediction_length:, 10:]

    x_test, y_test = (test[:, 0:arguments.prediction_length, 10:].reshape(test.shape[0], -1),
                      test[:, arguments.prediction_length:, 0:10]), test[:, arguments.prediction_length:, 10:]

    model = complexNDM(hidden_size=arguments.hidden_size,
                       output_size=arguments.output_size,
                       layer_num=arguments.layer_num,
                       sigma_min=arguments.r_min,
                       sigma_max=arguments.r_max,
                       scan=arguments.scan,
                       phase=arguments.phase)

    rng = jax.random.PRNGKey(arguments.seed)
    dummy_input = (jnp.ones((1, arguments.prediction_length * arguments.output_size)), jnp.ones((1, 128, 10)))
    params = model.init(rng, dummy_input)
    print(model.tabulate(rng, dummy_input))

    schedule = optax.schedules.warmup_cosine_decay_schedule(
      init_value=1e-7,
      peak_value=2e-4,
      warmup_steps=0.1*arguments.num_epochs*(len(dataset)//batch_size),
      decay_steps=arguments.num_epochs*(len(dataset)//batch_size),
      end_value=1e-7
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(schedule)
    )

    def loss_fn(params, x, y):
        y_pred, hidden_states = model.apply(params, x)

        loss_total = smoothl1loss(y, y_pred)
        loss_total = jnp.mean(loss_total)

        if arguments.is_PILF:
            diff = jnp.abs(hidden_states[0:-1] - hidden_states[1:])
            loss_smth = smoothl1loss(diff, jnp.zeros_like(diff))
            loss_smth = jnp.mean(loss_smth)

            ratio = jnp.divide(loss_smth, loss_total)
            ratio = jax.lax.stop_gradient(ratio)
            loss_total = loss_total + loss_smth / (10 * ratio)

        return loss_total

    @jax.jit
    def train_step(state, x, y):
        def loss(params):
            return loss_fn(params, x, y)

        loss, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    for epoch in range(arguments.num_epochs):
        epoch_loss_avg = 0.0
        print(f"Epoch {epoch + 1}/{arguments.num_epochs}\n---------------")
        with tqdm(train_data, desc="Training", unit="batch") as tqdm_bar:
          for i, (x1, x2, y) in enumerate(tqdm_bar):
              x1 = jnp.array(x1.numpy())
              x2 = jnp.array(x2.numpy())
              y = jnp.array(y.numpy())

              state, current_loss = train_step(state, (x1, x2), y)
              epoch_loss_avg += current_loss

              current_lr = schedule(state.step)
              tqdm_bar.set_postfix(loss=current_loss, lr=current_lr)

        epoch_loss_avg /= (i + 1)
        print("Epoch Avg Loss: {:.5f}".format(epoch_loss_avg))

        validations, _ = model.apply(state.params, x_valid)
        valid_loss = jnp.sqrt(jnp.mean(jnp.square(100 * validations - 100 * y_valid)))
        print("Valid Loss RMSE: {:.4f}".format(valid_loss) + '\n')

    predictions, _ = model.apply(state.params, x_test)
    test_loss = jnp.mean(jnp.square(100 * predictions - 100 * y_test))
    l_max = jnp.max(jnp.abs(100 * predictions - 100 * y_test))
    print("Test Loss MSE: {:.4f}".format(test_loss))
    print("Test Loss RMSE: {:.4f}".format(jnp.sqrt(test_loss)))
    print("Test Loss l_max: {:.4f}".format(l_max))

    bytes_output = flax.serialization.to_bytes(state.params)
    with open('checkpoints/best_model.flax', 'wb') as f:
        f.write(bytes_output)

    with open("exp.txt", "a") as file:
        content = (f"Seed: {arguments.seed}, r_min: {arguments.r_min}, r_max: {arguments.r_max}, "
                   f"phase: {arguments.phase:.3f}, MSE: {test_loss:.2f}, RMSE: {jnp.sqrt(test_loss):.2f}, "
                   f"l_max: {l_max:.2f}")
        file.write(content + "\n")


def main():
    args = parse_arguments()
    seed_random(args.seed)
    trainer(args)


if __name__ == '__main__':
    main()
