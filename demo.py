import os
import yaml
import imageio
import argparse
import numpy as np
import math
import pickle

from tqdm import tqdm
from smplx import SMPL
from einops import rearrange, repeat
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

from utils.features.kinetics import extract_kinetic_features
from utils.features.manual import extract_manual_features
from utils.metrics import *
from utils.renderer import get_renderer
from models.components.model import M2D
from datamodules.components.data_utils import get_audio_features, get_motion_features

import torch

Genres = {
    'gBR': 0,
    'gPO': 1,
    'gLO': 2,
    'gMH': 3,
    'gLH': 4,
    'gHO': 5,
    'gWA': 6,
    'gKR': 7,
    'gJS': 8,
    'gJB': 9,
}

# log_path: ./logs/runs/M2D_encoder_mintsetting # ckpt: path/checkpoints/last.ckpt
def load_model(log_path, ckpt):
    path = os.path.join('./logs', log_path)
    with open(os.path.join(path, '.hydra/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ckpt = torch.load(os.path.join(path, 'checkpoints', ckpt), map_location='cpu')
    state_dict = {}
    for key, value in ckpt['state_dict'].items():
        key = key.split('.')
        if key[0] == 'gen':
            state_dict.update({'.'.join(key[1:]): value})

    model = M2D(**config['model']['gen_params'])
    model.load_state_dict(state_dict)
    model.eval()

    return model

# pkl_data: 
def load_data(pkl_data, second, seed_m_length):
    pkl_data_path = os.path.join('../datasets/aistplusplus/motions', pkl_data)

    pose, trans = get_motion_features(pkl_data_path)
    motion = torch.cat([pose, trans], dim=1)

    audio_name = pkl_data.split('.')[0].split('_')[4]
    audio = get_audio_features(audio_name, '../datasets/aistplusplus')
    audio_path = os.path.join('../datasets/aistplusplus', 'original', audio_name + '.wav')

    genre_label = pkl_data.split('.')[0].split('_')[0]
    genre = torch.tensor(Genres[genre_label])

    audio = audio[:second * 60 + seed_m_length]
    seed_motion = motion[:seed_m_length]
    gt_motion = motion

    return audio, seed_motion, genre, audio_path, gt_motion

def calc_and_save_feats(root):
    
    # gt_list = []
    pred_list = []

    for pkl in os.listdir(root):
        print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:1200,:]
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 24))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # np_dance[:, :3] = root
        print(0)
        print(os.path.join(root, 'kinetic_features', pkl))
        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
        np.save(os.path.join(root, 'manual_features', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))

def calc_ba_score(root):

    # gt_list = []
    ba_scores = []

    # joint3d: only poses which passed smpl
    for pkl in os.listdir(root):
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()[:, :]

        dance_beats, length = calc_db(joint3d, pkl)        
        music_beats = get_mb(pkl.split('_')[4] + '.pkl', length)

        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

def save_features_render_video(motion, smpl, save_path, audio_path, pkl_data, device):
    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)

    for idx, motion_ in enumerate(motion):
        save_name = os.path.join(save_path, f'z{idx}.mp4')
        writer = imageio.get_writer(save_name, fps=60)

        pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
        
        # Smoothing
        smpl_poses = torch.from_numpy(smooth_pose(pose.cpu().numpy())).to(device)
        smpl_trans = torch.from_numpy(smooth_pose(trans.cpu().numpy())).to(device)

        # keypoints3d = smpl.forward(
        #     global_orient=smpl_poses[:, 0:1],
        #     body_pose=smpl_poses[:, 1:],
        #     transl=smpl_trans,
        # ).joints.detach().cpu().numpy()[:, 0:24, :]

        # roott = keypoints3d[:1, :1]  # the root
        # keypoints3d = keypoints3d - roott  # Calculate relative offset with respect to root
        # manual_features = extract_manual_features(keypoints3d)
        # os.makedirs(os.path.join(save_path, 'manual_features'), exist_ok=True)
        # np.save(os.path.join(save_path, 'manual_features', pkl_data.split('.')[0]+f'_manual_{idx}.npy'), manual_features)
        # kinetic_features = extract_kinetic_features(keypoints3d)
        # os.makedirs(os.path.join(save_path, 'kinetic_features'), exist_ok=True)
        # np.save(os.path.join(save_path, 'kinetic_features', pkl_data.split('.')[0]+f'_kinetic_{idx}.npy'), kinetic_features)
        # print('FID features saved!')

        meshes = smpl.forward(
            global_orient=smpl_poses[:, 0:1],
            body_pose=smpl_poses[:, 1:],
            transl=smpl_trans
        ).vertices.cpu().numpy()
        faces = smpl.faces

        meshes = meshes - meshes[0].mean(axis=0)
        cam = (0.55, 0.55, 0, 0.10)
        color = (0.2, 0.6, 1.0)

        imgs = []
        for ii, mesh in enumerate(tqdm(meshes, desc=f"Visualize dance - z{idx}")):
            img = renderer.render(background, mesh, faces, cam, color=color)
            imgs.append(img)

        imgs = np.array(imgs)
        for cimg in imgs:
            writer.append_data(cimg)
        writer.close()

        video_with_music(save_name, audio_path)

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0
    
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def smooth_pose(pred_pose, min_cutoff=0.004, beta=0.7):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.

    one_euro_filter = OneEuroFilter(
        np.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
    )

    pred_pose_hat = np.zeros_like(pred_pose)

    # initialize
    pred_pose_hat[0] = pred_pose[0]

    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1

        t = np.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose

    return np.array(pred_pose_hat)

def video_with_music(save_video, audio_path):
    videoclip = VideoFileClip(save_video)
    audioclip = AudioFileClip(audio_path)

    if os.path.isfile(save_video):
        os.remove(save_video)

    new_audioclip = CompositeAudioClip([audioclip])
    new_audioclip = new_audioclip.cutout(videoclip.duration, audioclip.duration)

    videoclip.audio = new_audioclip
    videoclip.write_videofile(save_video, logger=None)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.log_path, args.ckpt)
    audio, seed_motion, genre, audio_path, gt_motion = load_data(args.pkl_data, args.second, model.seed_m_length)

    smpl = SMPL(model_path='../datasets/smpl', gender='MALE', batch_size=1).eval()

    save_path = os.path.join('./logs/', args.log_path, 'demos', args.type)
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    smpl = smpl.to(device)

    if args.type == 'diversity':
        num_sample = 5
        noise = torch.randn(num_sample, 256).to(device)
        
        genre = repeat(genre[None], '() -> b', b=num_sample).to(device)

    else:
        num_sample = 1
        noise = torch.randn(1, 256).to(device)
        noise = repeat(noise, '() d -> b d', b=num_sample).to(device)
        genre = [idx for idx in range(10) if idx != genre]
        genre = torch.tensor(genre).long().to(device)

    audio = repeat(audio[None], '() n d -> b n d', b=num_sample).to(device)
    seed_motion = repeat(seed_motion[None], '() n d -> b n d', b=num_sample).to(device)
    gt_motion = repeat(gt_motion[None], '() n d -> b n d', b=num_sample).to(device)

    with torch.no_grad():
        output_motion = model.inference(audio, seed_motion, noise, genre)
        
        print('Calcuating and saving features and videos')
        save_features_render_video(output_motion, smpl, save_path, audio_path, args.pkl_data, device)
        # Compare with GT
        gt_path = os.path.join('./logs/', args.log_path, 'demos', args.type+'_gt')
        os.makedirs(gt_path, exist_ok=True)
        save_features_render_video(gt_motion, smpl, gt_path, audio_path, args.pkl_data, device)

        # print('Calculating and saving features')
        # calc_and_save_feats(save_path)
        # calc_and_save_feats(gt_path)

        # print('Calculating metrics')
        # print(save_path)
        # print(gt_path)
        # print(quantized_metrics(save_path, gt_path))
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="A Brand New Dance Partner")
    args.add_argument('-l', '--log_path', type=str, required=True)
    args.add_argument('-p', '--pkl_data', type=str, required=True)

    args.add_argument('-c', '--ckpt', type=str, default='last.ckpt')
    args.add_argument('-t', '--type', type=str, default='none')
    args.add_argument('-d', '--device', type=str, default='cuda:1')
    args.add_argument('-s', '--second', type=int, default=10)
    args = args.parse_args()

    main(args)
