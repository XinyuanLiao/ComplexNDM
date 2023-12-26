import utils.dataset.LoadData as ld
import utils.train.util as util


def build_dataset(cfg):
    #   DataSet
    train, valid, test = ld.loadData(cfg.dataset.estimate_window + cfg.dataset.predict_window, cfg.dataset.samples,
                                     cfg.dataset.DSample)
    train, valid, test = util.to_tensor(train).to(cfg.device), util.to_tensor(valid).to(cfg.device), \
        util.to_tensor(test).to(cfg.device)
    # print(train.shape, valid.shape, test.shape)
    valid_true = valid[:, cfg.dataset.estimate_window:, cfg.dataset.control:].permute(1, 0, 2)
    test_true = test[:, cfg.dataset.estimate_window:, cfg.dataset.control:].permute(1, 0, 2)
    raw_data = {
        "train": train,
        "valid": valid,
        "test": test,
        "test_true": test_true,
        "valid_true": valid_true
    }
    train_loader = ld.loader(train, cfg.dataset.estimate_window, cfg.train.batch, cfg.dataset.control)
    return train_loader, raw_data
