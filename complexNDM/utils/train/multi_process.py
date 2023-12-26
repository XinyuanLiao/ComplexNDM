import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import utils.train.adan as adan

SMAE = nn.SmoothL1Loss()
MSE = nn.MSELoss()

def mp_process_model_ddp(mp_args, h_step, world_size):
    dist.init_process_group("nccl", init_method="tcp://localhost:23456", rank=h_step, world_size=world_size)
    model_ddp = DistributedDataParallel(mp_args["model"])
    optimizer = adan.Adan(model_ddp.parameters(), lr=2e-4)
    output, h = model_ddp(mp_args["data"], mp_args["oc"], h_step)
    loss1 = SMAE(output, mp_args["label"][h_step].unsqueeze(0))
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()


def mp_process_model(mp_args):
    model = mp_args["model_mp"]
    optimizer_mp = adan.Adan(model.parameters(), lr=2e-4)
    model.to(mp_args["device"])

    # Forward data 
    output, _ = model(mp_args["data"], mp_args["oc"], mp_args["h_step"])
    loss1 = SMAE(output, mp_args["label"][mp_args["h_step"]].unsqueeze(0))

    # Backward pass
    optimizer_mp.zero_grad()
    loss1.backward()

    param_grads = []
    for param in model.parameters():
        if param.grad is not None:
            param_grads.append(param.grad.clone())
        else:
            param_grads.append(param.grad)
    return loss1.item(), param_grads

