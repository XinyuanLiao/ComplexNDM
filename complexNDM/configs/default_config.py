enable_mp = True
train = True
inference = True
plot = False
device = "cuda:0"

# train params
dataset = dict(
    DSample=8,
    samples=7000,
    estimate_window=16,
    predict_window=5,
    control=10,
)

train = dict(
    epochs=50000,
    batch=1024,
    num_process=2
)

model = dict(
    features=4,
    hidden_size=192,
    layers=3,
)

log_interval = 1

font = dict(
    family='Times New Roman',
    size=18
)
