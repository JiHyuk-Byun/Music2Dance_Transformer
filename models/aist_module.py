import torch
from utils.compute_loss import ComputeLoss
from models.components.model import M2D
from pytorch_lightning import LightningModule

class AISTLitModule(LightningModule):
    def __init__(self, gen_params, loss_params, optimizer, scheduler):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        assert gen_params, 'No generator parameters'
        self.gen = M2D(**gen_params)

        self.compute_loss = ComputeLoss(**loss_params)
    
    def gen_step(self, batch, state):
        dance_s, dance_f, music, dance_id = batch
        loss_op_dict = {'state': state,
                        'music': music,     
                        'dance_s': dance_s,     # motion
                        'dance_f': dance_f,     # loss 계산 위한 gt motion(next motion)
                        'dance_id': dance_id,   # genre
                        'gen': self.gen}    # 'gen' : M2D model

        return self.compute_loss(**loss_op_dict)
        
    def training_step(self, batch, batch_idx):
        gen_optim =  self.optimizers()

        gen_optim.zero_grad()
        loss, log_dict = self.gen_step(batch, "train_gen")

        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
        self.manual_backward(loss)
        gen_optim.step()

        if self.hparams.scheduler:
            gen_scheduler = self.lr_schedulers()
            gen_scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.gen_step(batch, "valid")
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer
        scheduler = self.hparams.scheduler

        gen_optim = getattr(torch.optim, optimizer['type'])(self.gen.parameters(), **optimizer['kwargs'])
        if scheduler:
            gen_scheduler = getattr(torch.optim.lr_scheduler, scheduler['type'])(gen_optim, **scheduler['kwargs'])
            return {"optimizer": gen_optim, "lr_scheduler": gen_scheduler}
        return gen_optim