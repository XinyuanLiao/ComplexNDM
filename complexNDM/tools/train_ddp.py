
import argparse
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import utils.train.adan as adan
from utils.dataset.build_dataset import build_dataset
from utils.model.build_model import build_model
import utils.train.util as util
from utils.train.multi_process import mp_process_model_ddp
from utils.fileio.config import Config

from utils.fileio.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train configs')
    parser.add_argument("--config", default="configs/default_config.py",help="train config file path")
    parser.add_argument('--seed', type=int, default=920, help='Random seed')
    parser.add_argument('--enable_mp', action='store_true')

    args = parser.parse_args()
    return args

def train(cfg):
    config = {
      "font.family": cfg.font.family,  
      "font.size": cfg.font.size,
    }
    rcParams.update(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, raw_data = build_dataset(cfg)
    model = build_model(cfg)

    if cfg.enable_mp:
        print("Start Multi-Process Training")
        mp.set_start_method("spawn")
    else:
        print("Start Sinlge-Process Training")
    
    T_loss, V_loss = [], []
    # Train
    if cfg.train:
        optimizer = adan.Adan(model.parameters(), lr=2e-4)
        early_stop = util.EarlyStopping(patience=20, cold=3, path=f'./checkpoint/complexNDM.pth')
        for epoch in range(cfg.train.epochs):
            start_time = time.time()
            for _, (oc, data, label) in enumerate(train_loader):
                label = label.permute(1, 0, 2)
                if cfg.enable_mp:
                    mp_args = {
                        "model": model,
                        "data": data,
                        "label": label,
                        "oc": oc,
                        "device": device
                    }
                    processes = []
                    for h_step in range(cfg.train.num_process):
                        p = mp.Process(target = mp_process_model_ddp,  args = (mp_args, h_step, cfg.train.num_process))
                        processes.append(p)
                        p.start()
                    for p in processes:
                        p.join()
                else:
                    output, h = model(data, oc)
                    loss1 = SMAE(label, output)
                    loss2 = 0.1 * SMAE(torch.abs(h[:-1, :, :]), torch.abs(h[1:, :, :]))
                    loss = loss1 + loss2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch} took {epoch_time:.2f} seconds.")
            with torch.no_grad():
                val_o, _ = model(raw_data["valid"][:, 0:cfg.dataset.estimate_window, cfg.dataset.control:],
                                 raw_data["valid"][:, cfg.dataset.estimate_window:, 0:cfg.dataset.control])
                val_loss = SMAE(val_o, raw_data["valid_true"])
                vl1 = MSE(100 * val_o, 100 * raw_data["valid_true"])

            early_stop(val_loss, model, optimizer)

            if epoch % cfg.log_interval == 0:
                print('epoch:%-5d train_loss: %-12.3e smooth_loss: %-12.3e valid_loss: %-9.3f '
                    'learning rate: %.1e' %
                    (epoch, util.to_numpy(loss1), util.to_numpy(loss2), util.to_numpy(vl1),
                    optimizer.param_groups[0]['lr']))
            if early_stop.early_stop:
                break

    dist.destroy_process_group()
    util.save_losses(V_loss, f'./test/complex_valid.txt')
    util.save_losses(T_loss, f'./test/complex_train.txt')

    # Test
    if args.predict:
        model.load_state_dict(torch.load(f'./checkpoint/complexNDM.pth'))
        # print(torch.diag(model.effective_W()))
        test_pred, _ = model(raw_data["test"][:, 0:cfg.dataset.estimate_window, cfg.dataset.control:],
                             raw_data["test"][:, cfg.dataset.estimate_window:, 0:cfg.dataset.control])
        test_loss = MSE(100 * test_pred, 100 * raw_data["test_true"])
        # with open("complexNDM.txt", 'a') as file:
        #     file.write(str(args.hidden_size) + '\t' + str(util.to_numpy(test_loss)) + '\n')
        print('Test Loss: %.5f' % test_loss)
        print('\n')

    if cfg.plot:
        model.load_state_dict(torch.load(f'./checkpoint/complexNDM.pth'))
        plot(model)

def plot(model):
    eigens = util.to_numpy(model.w_effect())
    fig, ax = plt.subplots(dpi=100)
    circle = patches.Circle((0, 0), 1, fill=False, color='#CE6A6C', linewidth=3)
    ax.add_patch(circle)
    x = np.diag(eigens.real)
    y = np.diag(eigens.imag)
    plt.scatter(x, y, color='#2278B4', s=10)
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.enable_mp = cfg.enable_mp or args.enable_mp

    util.seed_torch(args.seed)
    train(cfg)

if __name__ == '__main__':
    main()
