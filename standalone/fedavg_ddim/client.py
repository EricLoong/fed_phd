
class client:
    def __init__(self, client_idx, train_data_local_number,args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_sample_number = train_data_local_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def train(self, w_global, round_idx):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.train()
        weights = self.model_trainer.get_model_params()
        return weights

    def get_sample_number(self):
        return self.local_sample_number








