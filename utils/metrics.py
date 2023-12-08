import numpy as np
import pickle 
from utils.features.kinetics import extract_kinetic_features
from utils.features.manual import extract_manual_features
from scipy import linalg

# kinetic, manual
import os

from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt 

music_root = '/workspace/ssd1/users/jhbyun/datasets/aistplusplus/model_seperate_features'

def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path, 'rb') as f:
        #print(path)
        pkl = pickle.load(f)
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

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def quantized_metrics(predicted_pkl_root, gt_pkl_root):


    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []


    # for pkl in os.listdir(predicted_pkl_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features'))]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 

#   T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize(gt_freatures_k)
    # gt_freatures_m = normalize(gt_freatures_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
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
        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
        np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))


if __name__ == '__main__':


    gt_root = 'data/aist_features_zero_start'
    pred_root = 'experiments/actor_critic/eval/pkl/ep000010'
    print('Calculating and saving features')
    calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)


    print('Calculating metrics')
    print(gt_root)
    print(pred_root)
    print(quantized_metrics(pred_root, gt_root))