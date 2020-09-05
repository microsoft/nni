import nni

class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def bind_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def configure_optimizer(self):
        raise NotImplementedError()

    def train_step(self, x, y, infer_y):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def training_logic(self):
        # TODO: model device and data device
        self.model.cuda()
        for _ in range(self.n_epochs):
            self.train()
            test_acc = self.validate()
            nni.report_intermediate_result(test_acc)
        nni.report_final_result(test_acc)