from utils.model.models import complexNDM
import utils.train.util as util

def build_model(cfg, is_cpu=False):
    if is_cpu:
        device = "cpu"
    else:
        device = cfg.device
    model = complexNDM(cfg.dataset.control, cfg.dataset.estimate_window, 
                      cfg.model.hidden_size, cfg.model.features, 
                      cfg.model.layers, cfg.enable_mp).to(device)
    # print(f'Total parameters of complexNDM is {util.get_parameters(model)}')
    return model
    