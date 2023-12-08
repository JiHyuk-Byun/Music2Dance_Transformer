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
from demo import *

import torch

def fid_evaluation(motion, smpl, save_path, pkl_data, device):

    for idx, motion_ in enumerate(motion):
        pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
            
        # Smoothing
        smpl_poses = torch.from_numpy(smooth_pose(pose.cpu().numpy())).to(device)
        smpl_trans = torch.from_numpy(smooth_pose(trans.cpu().numpy())).to(device)

        keypoints3d = smpl.forward(
            global_orient=smpl_poses[:, 0:1],
            body_pose=smpl_poses[:, 1:],
            transl=smpl_trans,
        ).joints.detach().cpu().numpy()[:, 0:24, :]

        roott = keypoints3d[:1, :1]  # the root
        keypoints3d = keypoints3d - roott  # Calculate relative offset with respect to root
        manual_features = extract_manual_features(keypoints3d)
        os.makedirs(os.path.join(save_path, 'manual_features'), exist_ok=True)
        np.save(os.path.join(save_path, 'manual_features', pkl_data.split('.')[0]+f'_manual_{idx}.npy'), manual_features)
        kinetic_features = extract_kinetic_features(keypoints3d)
        os.makedirs(os.path.join(save_path, 'kinetic_features'), exist_ok=True)
        np.save(os.path.join(save_path, 'kinetic_features', pkl_data.split('.')[0]+f'_kinetic_{idx}.npy'), kinetic_features)
        print('FID features saved!')

def beatalign_evaluation(motion, smpl, device):
    for idx, motion_ in enumerate(motion):
        pose, trans = motion_[:, :-3].view(-1, 24, 3), motion_[:, -3:]
            
        # Smoothing
        smpl_poses = torch.from_numpy(smooth_pose(pose.cpu().numpy())).to(device)
        smpl_trans = torch.from_numpy(smooth_pose(trans.cpu().numpy())).to(device)

        joint3d = smpl.forward(
            global_orient=smpl_poses[:, 0:1],
            body_pose=smpl_poses[:, 1:],
            transl=smpl_trans,
        ).joints.detach().cpu().numpy()[:, 0:24, :]

        dance_beats, length = calc_db(joint3d)
    
    return dance_beats, length

def main_fid(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.log_path, args.ckpt)
    smpl = SMPL(model_path='../datasets/smpl', gender='MALE', batch_size=1).eval()

    save_path = './predicted_FID_features'
    gt_path = './gt_FID_features'
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    smpl = smpl.to(device)
    noise = torch.randn(1, 256).to(device)

    test_data_list = ['gJB_sFM_cAll_d07_mJB4_ch05', 'gPO_sBM_cAll_d10_mPO1_ch10', 'gKR_sBM_cAll_d29_mKR5_ch09', 'gHO_sBM_cAll_d20_mHO1_ch01', 'gMH_sBM_cAll_d22_mMH2_ch10', 'gLO_sBM_cAll_d15_mLO5_ch05', 'gLO_sBM_cAll_d13_mLO2_ch07', 'gJS_sBM_cAll_d01_mJS1_ch05', 'gJB_sBM_cAll_d09_mJB2_ch05', 'gLH_sBM_cAll_d16_mLH0_ch06', 'gBR_sBM_cAll_d06_mBR3_ch08', 'gBR_sBM_cAll_d06_mBR3_ch10', 'gHO_sBM_cAll_d21_mHO5_ch01', 'gBR_sFM_cAll_d06_mBR4_ch20', 'gMH_sBM_cAll_d22_mMH3_ch10', 'gPO_sBM_cAll_d11_mPO4_ch07', 'gPO_sBM_cAll_d11_mPO4_ch02', 'gHO_sBM_cAll_d21_mHO2_ch03', 'gLH_sBM_cAll_d17_mLH5_ch03', 'gJS_sBM_cAll_d03_mJS3_ch02', 'gHO_sBM_cAll_d21_mHO5_ch07', 'gBR_sBM_cAll_d06_mBR4_ch03', 'gBR_sBM_cAll_d05_mBR0_ch08', 'gWA_sBM_cAll_d25_mWA1_ch06', 'gMH_sBM_cAll_d24_mMH4_ch05', 'gWA_sBM_cAll_d26_mWA5_ch08', 'gWA_sBM_cAll_d27_mWA3_ch09', 'gKR_sBM_cAll_d30_mKR3_ch06', 'gWA_sBM_cAll_d27_mWA2_ch07', 'gLO_sBM_cAll_d13_mLO1_ch03']
    for pkl_data in test_data_list:
        print(f'Processing {pkl_data}...')
        audio, seed_motion, genre, audio_path, gt_motion = load_data(pkl_data+'.pkl', args.second, model.seed_m_length)
        audio, seed_motion, gt_motion, genre = audio[None].to(device), seed_motion[None].to(device), gt_motion[None].to(device), genre[None].to(device)

        with torch.no_grad():
            output_motion = model.inference(audio, seed_motion, noise, genre)
            fid_evaluation(output_motion, smpl, save_path, pkl_data+'.pkl', device)
            fid_evaluation(gt_motion, smpl, gt_path, pkl_data+'.pkl', device)
    
    print(quantized_metrics(save_path, gt_path))

def main_beat(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.log_path, args.ckpt)
    smpl = SMPL(model_path='../datasets/smpl', gender='MALE', batch_size=1).eval()

    model = model.to(device)
    smpl = smpl.to(device)
    noise = torch.randn(1, 256).to(device)
    ba_scores = []

    test_data_list = ['gJB_sFM_cAll_d07_mJB4_ch05', 'gPO_sBM_cAll_d10_mPO1_ch10', 'gKR_sBM_cAll_d29_mKR5_ch09', 'gHO_sBM_cAll_d20_mHO1_ch01', 'gMH_sBM_cAll_d22_mMH2_ch10', 'gLO_sBM_cAll_d15_mLO5_ch05', 'gLO_sBM_cAll_d13_mLO2_ch07', 'gJS_sBM_cAll_d01_mJS1_ch05', 'gJB_sBM_cAll_d09_mJB2_ch05', 'gLH_sBM_cAll_d16_mLH0_ch06', 'gBR_sBM_cAll_d06_mBR3_ch08', 'gBR_sBM_cAll_d06_mBR3_ch10', 'gHO_sBM_cAll_d21_mHO5_ch01', 'gBR_sFM_cAll_d06_mBR4_ch20', 'gMH_sBM_cAll_d22_mMH3_ch10', 'gPO_sBM_cAll_d11_mPO4_ch07', 'gPO_sBM_cAll_d11_mPO4_ch02', 'gHO_sBM_cAll_d21_mHO2_ch03', 'gLH_sBM_cAll_d17_mLH5_ch03', 'gJS_sBM_cAll_d03_mJS3_ch02', 'gHO_sBM_cAll_d21_mHO5_ch07', 'gBR_sBM_cAll_d06_mBR4_ch03', 'gBR_sBM_cAll_d05_mBR0_ch08', 'gWA_sBM_cAll_d25_mWA1_ch06', 'gMH_sBM_cAll_d24_mMH4_ch05', 'gWA_sBM_cAll_d26_mWA5_ch08', 'gWA_sBM_cAll_d27_mWA3_ch09', 'gKR_sBM_cAll_d30_mKR3_ch06', 'gWA_sBM_cAll_d27_mWA2_ch07', 'gLO_sBM_cAll_d13_mLO1_ch03']
    for pkl_data in test_data_list:
        print(f'Processing {pkl_data}...')
        audio, seed_motion, genre, audio_path, gt_motion = load_data(pkl_data+'.pkl', args.second, model.seed_m_length)
        audio, seed_motion, gt_motion, genre = audio[None].to(device), seed_motion[None].to(device), gt_motion[None].to(device), genre[None].to(device)

        with torch.no_grad():
            output_motion = model.inference(audio, seed_motion, noise, genre)
            dance_beats, length = beatalign_evaluation(output_motion, smpl, device)
            music_beats = get_mb(pkl_data.split('_')[4] + '.pkl', length)
            ba_scores.append(BA(music_beats, dance_beats))
    print(len(ba_scores))
    print(np.mean(ba_scores))

    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description="A Brand New Dance Partner")
    args.add_argument('-l', '--log_path', type=str, required=True)
    args.add_argument('-c', '--ckpt', type=str, default='last.ckpt')
    args.add_argument('-d', '--device', type=str, default='cuda:0')
    args.add_argument('-s', '--second', type=int, default=20)
    args = args.parse_args()
    # print(quantized_metrics('./predicted_FID_features', './gt_FID_features'))
    # main_fid(args)
    main_beat(args)