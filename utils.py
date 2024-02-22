class Early_Stop_Callback:
    def __init__(self, model, patience=20, factor=0.5, min_lr=0, repeat=3):
        self.model = model
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.repeat = repeat
        self.wait = 0
        self.num_repeats = 0
        self.best_val_loss = float('inf')
        self.lr = 0
        self.stop_training = False

    def __call__(self, current_val_loss):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
            self.model.save_weights('./checkpoints/best_model')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.num_repeats += 1
                if self.num_repeats >= self.repeat:
                    print('Reached maximum repeats, stopping training.')
                    self.stop_training = True
                else:
                    self.wait = 0
                    self.lr = max(self.lr * self.factor, self.min_lr)
                    self.model.optimizer.lr = self.lr
                    print(f'Reducing learning rate to {self.lr} and resetting patience.')
                    self.wait = 0
