import torch
from pytorch_lightning import LightningModule
from smplx import SMPL
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import wandb

from utils.compute_loss import ComputeLoss
from models.components.model import M2D
from utils.renderer import get_renderer


class AISTLitModule(LightningModule):
    def __init__(self, gen_params, loss_params, optimizer, scheduler):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        assert gen_params, 'No generator parameters'
        self.gen = M2D(**gen_params)

        self.compute_loss = ComputeLoss(**loss_params)
        self.n_samples = 4

        # self.smpl = SMPL(model_path=gen_params['smpl'], gender='MALE', batch_size=1).eval()
        # self.smpl.to(device)


    def gen_step(self, batch, state):
        dance_s, dance_f, music, dance_id = batch
        loss_op_dict = {'state': state,
                        'music': music,     
                        'dance_s': dance_s,     # motion
                        'dance_f': dance_f,     # loss 계산 위한 gt motion(next motion)
                        'dance_id': dance_id,   # genre
                        'gen': self.gen}    # 'gen' : M2D model

        return self.compute_loss(**loss_op_dict) # gen_loss: loss, log_dict
        
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
    
    def test_step(self, batch, batch_idx):
        
        loss, log_dict = self.gen_step(batch, "valid")
        self.log_dict(log_dict, on_step=True, on_epoch=False, prog_bar=False)
        
        # if batch_idx == 0:
            
        #     wandb_videos_pred = list()
        #     for video_idx in range(self.samples):
        #         dance_s, dance_f, music, genre = batch

        #         noise = torch.randn(, 256).to(device)
        #         pred_motions = self.gen.inference(audio=batch[2][video_idx],
        #                                           motion=batch[0][video_idx], 
        #                                           noise=noise, 
        #                                           genre=batch[3][video_idx])
        #         pred_video = render_video(pred_motion, self.smpl)

        #         wandb_videos_pred.append(pred_video)
        #     self.logger.log_video(key="Rendering on Validation Set(Predict)", videos=wandb_videos_pred)

        return loss
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer
        scheduler = self.hparams.scheduler

        gen_optim = getattr(torch.optim, optimizer['type'])(self.gen.parameters(), **optimizer['kwargs'])
        if scheduler:
            gen_scheduler = getattr(torch.optim.lr_scheduler, scheduler['type'])(gen_optim, **scheduler['kwargs'])
            return {"optimizer": gen_optim, "lr_scheduler": gen_scheduler}
        return gen_optim

    # def render_video(motion, smpl):
    #     width = 1024
    #     height = 1024

    #     background = np.zeros((height, width, 3))
    #     # 'background': [1024, 1024, 3]
    #     renderer = get_renderer(width, height)

    #     for idx, motion_ in enumerate(motion):
    #         #save_name = os.path.join(save_path, f'z{idx}.mp4')
    #         #writer = imageio.get_writer(save_name, fps=60)

    #         pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
    #         meshes = smpl.forward(
    #             global_orient=pose[:, 0:1],
    #             body_pose=pose[:, 1:],
    #             transl=trans
    #         ).vertices.cpu().numpy()
    #         faces = smpl.faces

    #         meshes = meshes - meshes[0].mean(axis=0)
    #         cam = (0.55, 0.55, 0, 0.10)
    #         color = (0.2, 0.6, 1.0)

    #         # video
    #         imgs = []
    #         for ii, mesh in enumerate(tqdm(meshes, desc=f"Visualize dance - z{idx}")):
    #             img = renderer.render(background, mesh, faces, cam, color=color)
    #             imgs.append(img)

    #         imgs = np.array(imgs)
    #         # 'imgs: [time, 3, 1024, 1024], fps = 60

    #         return imgs
    #         # for cimg in imgs:
    #         #     writer.append_data(cimg)
    #         # writer.close()

    #         # video_with_music(save_name, audio_path)

    # # def video_with_music(imgs, audio_path):
    # #     videoclip = VideoFileClip(save_video)
    # #     audioclip = AudioFileClip(audio_path)

    # #     if os.path.isfile(save_video):
    # #         os.remove(save_video)

    # #     new_audioclip = CompositeAudioClip([audioclip])
    # #     new_audioclip = new_audioclip.cutout(videoclip.duration, audioclip.duration)

    # #     videoclip.audio = new_audioclip
    # #     videoclip.write_videofile(save_video, logger=None)
