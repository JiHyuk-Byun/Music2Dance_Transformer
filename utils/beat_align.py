import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt 

music_root = '/workspace/ssd1/users/jhbyun/datasets/aistplusplus/model_separate_features'


def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        pkl = np.loads(f.read())
        if length is not None:
            beats = np.array(pkl)[:, 385][:][:length]
        else:
            beats = np.array(pkl)[:, 385]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')


        # print(len(beats))
        return beat_axis

# keypoints: only poses
def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

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

if __name__ == '__main__':

    pred_root = '/mnt/lustressd/lisiyao1/dance_experiements/experiments/music_ccgpt2_ac_ba_1e-5_freeze_droupout_beta0.5/vis/pkl/ep000005'

    print(calc_ba_score(pred_root))