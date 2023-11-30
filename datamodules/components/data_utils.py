import os
import random
import librosa

import numpy as np
import pickle as pkl
import torch

#from src.datamodules.components._preprocess_wav import FeatureExtractor
#from src.utils.rotation_conversions import *


dance_genre_dict = {
    "gBR": 0,
    "gPO": 1,
    "gLO": 2,
    "gMH": 3,
    "gLH": 4,
    "gHO": 5,
    "gWA": 6,
    "gKR": 7,
    "gJS": 8,
    "gJB": 9
}

# extractor = FeatureExtractor()
def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# 시작 index sampling
def get_feature_sample(total_length, sample_length):
    return random.randint(0, total_length - sample_length)
    #
    # sample_idx = [pivot + i for i in range(sample_length)]
    # return sample_idx

# .pkl로 저장된 feature를 받아서 k: 음악이름, v: feature dictionary로 반환.
# 'feature_path': .pkl 폴더, 'src_dir': aistplusplus 폴더
def processing_music_list(src_dir):
    music_dict = {}
    feature_dir = os.path.join(src_dir, 'model_seperate_features')
    wav_dir = os.path.join(src_dir, 'original')
    
    makedirs(feature_dir)
    makedirs(wav_dir)
    
    music_name = set([file.split('.')[0] for file in os.listdir(wav_dir)])

    for name in music_name:
        feature_path = os.path.join(feature_dir, f'{name}.pkl')
        if os.path.exists(feature_path):
            music = np.load(feature_path, allow_pickle=True)
            # print("music feature:", music.shape)
            music_dict.update({name: torch.from_numpy(music).float()})
        else:
            music = get_audio_features(name, src_dir)
            music_dict.update({name: music})
    return music_dict

# .pkl로 저장된 feature를 받아서 k: motion name, v: feature dictionary + motion name list 반환.
def processing_dance_list(src_dir):
    dance_dict = {}
    dance_list = []
    motion_dir = os.path.join(src_dir, 'motions') 
    for pkl_file in os.listdir(motion_dir):
        pkl_path = os.path.join(motion_dir, pkl_file)
        dance = get_motion_features(pkl_path)
        dance_dict.update({pkl_file.split('.')[0]: dance})
        dance_list.append(pkl_file.split('.')[0])
    return dance_dict, dance_list

# music feature extraction
def get_audio_features(audio_name, src_dir):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(drum_name):
        """Get tempo (BPM) for a music by parsing music name."""
        audio_name = drum_name.split('_')[1]
        assert len(audio_name) == 4

        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else: assert False, audio_name

    # audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))

    # File name
    print("#" * 5 + "Extracting features of {}...".format(audio_name) + "#" * 5)
    original_path = os.path.join(src_dir, "original", f"{audio_name}.wav")
    drum_path = os.path.join(src_dir, "model_seperate_wav", f"{audio_name}_drums.wav")
    melody_path = os.path.join(src_dir, "model_seperate_wav", f"{audio_name}_other.wav")

    save_path = os.path.join(src_dir, "model_seperate_features", f"{audio_name}.pkl")

    # load wav data onto librosa
    data, _ = librosa.load(original_path, sr=SR)
    data_drum, _ = librosa.load(drum_path, sr=SR)
    data_melody, _ = librosa.load(melody_path, sr=SR)

    ## DRUM ##
    # envelope
    drum_env = librosa.onset.onset_strength(y=data_drum, sr=SR)  # (seq_len,)            
    # tempogram
    tempogram = librosa.feature.tempogram(y=data_drum, sr=SR, onset_envelope=drum_env, hop_length=HOP_LENGTH).T  # (seq_len, 20)
    # peak
    drum_peak_idxs = librosa.onset.onset_detect(
        onset_envelope=drum_env.flatten(), sr=SR, hop_length=HOP_LENGTH)
    drum_peak_onehot = np.zeros_like(drum_env, dtype=np.float32)
    drum_peak_onehot[drum_peak_idxs] = 1.0  # (seq_len,)            
    # beats
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=drum_env, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=_get_tempo("drum_" + audio_name), tightness=100)
    beat_onehot = np.zeros_like(drum_env, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)     
    print("Drum feature extracted!!")

    ## MELODY ##
    # MFCC
    melody_mfcc = librosa.feature.mfcc(y=data_melody, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    # envelope
    melody_env = librosa.onset.onset_strength(y=data_melody, sr=SR)  # (seq_len,)                    
    # peak
    melody_peak_idxs = librosa.onset.onset_detect(
        onset_envelope=melody_env.flatten(), sr=SR, hop_length=HOP_LENGTH)
    melody_peak_onehot = np.zeros_like(melody_env, dtype=np.float32)
    melody_peak_onehot[melody_peak_idxs] = 1.0  # (seq_len,)
    # chroma
    melody_chroma = librosa.feature.chroma_cens(y=data_melody, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T          
    
    print("MELODY feature extracted!!")

    ## ORIGINAL ##
    origin_mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)            
    print("Origin feature extracted!!")


    # Padding to make same length for max size feature
    max_size = max([arr.shape[0] for arr in [drum_env, tempogram, drum_peak_onehot, beat_onehot,
                                        melody_mfcc, melody_env, melody_peak_onehot, melody_chroma,
                                        origin_mfcc]])
    padding_func = lambda arr: np.pad(arr, ((0,max_size-arr.shape[0]), (0,0)), mode='constant')

    drum_env_padded = padding_func(drum_env[:,None])
    tempogram_padded = padding_func(tempogram)
    drum_peak_onehot_padded = padding_func(drum_peak_onehot[:,None])
    beat_onehot_padded = padding_func(beat_onehot[:,None])
    melody_mfcc_padded = padding_func(melody_mfcc)
    melody_env_padded = padding_func(melody_env[:,None])
    melody_peak_onehot_padded = padding_func(melody_peak_onehot[:,None])
    melody_chroma_padded = padding_func(melody_chroma)
    origin_mfcc_padded = padding_func(origin_mfcc)

    audio_feature = np.concatenate([
        drum_env_padded,         # 1
        tempogram_padded,        # 384
        drum_peak_onehot_padded, # 1
        beat_onehot_padded,      # 1
        melody_mfcc_padded,       # 20
        melody_env_padded,        # 1
        melody_peak_onehot_padded,# 1
        melody_chroma_padded,     # 12
        origin_mfcc_padded      # 20
        ], axis=-1) # 441

    # Save feature as .pkl
    print("Saving {}...".format(audio_name))     
    makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as f:  
        pkl.dump(audio_feature, f)
    
    audio_feature = torch.from_numpy(audio_feature).float()

    return audio_feature

# pkl로 저장된 motion feature를 torch tensor로 변환하여 upload
def get_motion_features(pkl_path):
    motion = pkl.load(open(pkl_path, "rb"))

    smpl_poses = torch.from_numpy(motion['smpl_poses']).float() #  [length, 3]
    smpl_trans = torch.from_numpy(motion['smpl_trans'] / motion['smpl_scaling']).float() # [length, 72]

    # print("smpl_poses: ", smpl_poses.shape)
    # print("smpl_trans: ", smpl_trans.shape)

    # ret = torch.cat([smpl_poses, smpl_trans], dim=1)
    # ret = smpl_poses
    return [smpl_poses, smpl_trans]

# def wav_processing(wav_path, audio_name):
#     FPS = 60
#     HOP_LENGTH = 512
#     SR = FPS * HOP_LENGTH
#     EPS = 1e-6

#     def _get_tempo(audio_name):
#         """Get tempo (BPM) for a music by parsing music name."""
#         assert len(audio_name) == 4
#         if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
#             return int(audio_name[3]) * 10 + 80
#         elif audio_name[0:3] == 'mHO':
#             return int(audio_name[3]) * 5 + 110
#         else:
#             assert False, audio_name

#     audio, _ = librosa.load(wav_path, sr=SR)
#     melspe_db = extractor.get_melspectrogram(audio, SR)

#     mfcc = extractor.get_mfcc(melspe_db)
#     mfcc_delta = extractor.get_mfcc_delta(mfcc)
#     # mfcc_delta2 = get_mfcc_delta2(mfcc)

#     audio_harmonic, audio_percussive = extractor.get_hpss(audio)
#     # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
#     # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
#     chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave=7 if SR == 15360 * 2 else 5)
#     # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

#     onset_env = extractor.get_onset_strength(audio_percussive, SR)
#     tempogram = extractor.get_tempogram(onset_env, SR)
#     onset_beat = extractor.get_onset_beat(onset_env, SR)[0]
#     # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
#     # onset_beats.append(onset_beat)

#     onset_env = onset_env.reshape(1, -1)

#     feature = np.concatenate([
#         # melspe_db,
#         mfcc,  # 20
#         mfcc_delta,  # 20
#         # mfcc_delta2,
#         # harmonic_melspe_db,
#         # percussive_melspe_db,
#         # chroma_stft,
#         chroma_cqt,  # 12
#         onset_env,  # 1
#         onset_beat,  # 1
#         tempogram
#     ], axis=0)

#     # mfcc, #20
#     # mfcc_delta, #20

#     # chroma_cqt, #12
#     # onset_env, # 1
#     # onset_beat, #1

#     feature = feature.transpose(1, 0)

#     save_path = wav_path.split('.')[0]
#     np.save(f'{save_path}.npy', feature)
#     audio_feature = torch.from_numpy(feature).float()
#     return audio_feature