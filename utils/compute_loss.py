import torch
import torch.nn as nn

from smplx import SMPL
from einops import rearrange
import utils.rotation_conversions as geometry 

class ComputeLoss(nn.Module):
    def __init__(self, rot_6d):
        super(ComputeLoss, self).__init__()
        self.rot_6d = rot_6d

        self.l2_loss = nn.MSELoss()

    def __call__(self, **kwargs):
        state = kwargs['state']
        if state in ["train_gen", "valid"]:
            return self.gen_loss(**kwargs)
        
    def gen_loss(self, **kwargs):
        state = kwargs['state']
        music, dance_s, dance_f, dance_id = kwargs['music'], kwargs['dance_s'], kwargs['dance_f'], kwargs['dance_id']

        b = music.shape[0]  # 'b' : batch size
        device = music.device

        noise = torch.randn(b, 256).to(device)
        dance_g = kwargs['gen'](music, dance_s, noise, dance_id)    # predicted motion

        if self.rot_6d:
            dance_f = geometry.matTOrot6d(dance_f)

        loss, log_dict = 0.0, {}

        # Computing MSE loss
        mse_loss = self.l2_loss(dance_f, dance_g)
        log_dict.update({f'{state}/mse_loss': mse_loss})

        # Computing Sequential loss
        gt_motion_diff = dance_f[:, 1:] - dance_f[:, :-1]
        pred_motion_diff = dance_g[:, 1:] - dance_g[:, :-1]
        seq_loss = self.l2_loss(gt_motion_diff, pred_motion_diff)
        log_dict.update({f'{state}/seq_loss': seq_loss})

        # Compute final loss
        loss = 0.636 * mse_loss + 2.964 * seq_loss
        log_dict.update({f'{state}/loss': loss})

        return loss, log_dict


