import numpy as np # linear algebra
import os
import librosa
import pickle

# AIST(wav) -> drum(midi)   -> Tempogram, 1-dim envelope, 1-dim one-hot peaks 1-dim one-hot beats    
#           -> melody(midi) -> MFCC

#Return 
'''
Input       Output
'Original': MFCC, chroma
'piano':    MFCC, envelope, one-hot peaks
'drum':     Tempogram, envelope, one-hot peaks, one-hot beats, 
MFCC Tempogram, 1-dim envelope, 1-dim one-hot peaks, 1-dim one-hot beats
'''
def cache_audio_features(audio_names, save_dir, src_dir, audio_dir):
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

    for audio_name in audio_names:
        # File name
        print("#" * 5 + "Extracting features of {}...".format(audio_name) + "#" * 5)
        original_dir = os.path.join(audio_dir , f"{audio_name}.wav")
        drum_dir = os.path.join(src_dir , f"{audio_name}_drums.wav")
        piano_dir = os.path.join(src_dir , f"{audio_name}_other.wav")

        save_path = os.path.join(save_dir, f"{audio_name}.npy")

        # if os.path.exists(save_path):
        #     continue
        print(drum_dir)
        data, _ = librosa.load(original_dir, sr=SR)
        data_drum, _ = librosa.load(drum_dir, sr=SR)
        data_piano, _ = librosa.load(piano_dir, sr=SR)

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

        ## PIANO ##
        # MFCC
        piano_mfcc = librosa.feature.mfcc(y=data_piano, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        # envelope
        piano_env = librosa.onset.onset_strength(y=data_piano, sr=SR)  # (seq_len,)                    
        # peak
        piano_peak_idxs = librosa.onset.onset_detect(
            onset_envelope=piano_env.flatten(), sr=SR, hop_length=HOP_LENGTH)
        piano_peak_onehot = np.zeros_like(piano_env, dtype=np.float32)
        piano_peak_onehot[piano_peak_idxs] = 1.0  # (seq_len,)            
        print("PIANO feature extracted!!")

        ## ORIGINAL ##
        origin_mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)            
        origin_chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T          
        print("Origin feature extracted!!")


        # Padding to make same length for max size feature
        max_size = max([arr.shape[0] for arr in [drum_env, tempogram, drum_peak_onehot, beat_onehot,
                                          piano_mfcc, piano_env, piano_peak_onehot,
                                          origin_mfcc, origin_chroma]])
        padding_func = lambda arr: np.pad(arr, ((0,max_size-arr.shape[0]), (0,0)), mode='constant')

        drum_env_padded = padding_func(drum_env[:,None])
        tempogram_padded = padding_func(tempogram)
        drum_peak_onehot_padded = padding_func(drum_peak_onehot[:,None])
        beat_onehot_padded = padding_func(beat_onehot[:,None])
        piano_mfcc_padded = padding_func(piano_mfcc)
        piano_env_padded = padding_func(piano_env[:,None])
        piano_peak_onehot_padded = padding_func(piano_peak_onehot[:,None])
        origin_mfcc_padded = padding_func(origin_mfcc)
        origin_chroma_padded = padding_func(origin_chroma)


        audio_feature = np.concatenate([drum_env_padded, tempogram_padded, drum_peak_onehot_padded, beat_onehot_padded,
                                        piano_mfcc_padded, piano_env_padded, piano_peak_onehot_padded,
                                        origin_mfcc_padded, origin_chroma_padded], axis=-1)

        # Save feature as .pkl
        print("Saving {}...".format(audio_name))     
        save_file = os.path.join(save_dir, audio_name)
        with open(save_file +'.pkl', 'wb') as f:  
            pickle.dump(audio_feature, f)

if __name__ == '__main__':
    base_dir = '/workspace/ssd1/users/jhbyun/datasets/aistplusplus/'
    src_dir = base_dir + 'model_seperate_wav/'
    audio_dir = base_dir + 'wav/original/'
    audio_names = os.listdir(audio_dir)
    audio_names = list(audio_name.split('.')[0] for audio_name in audio_names)
    save_dir = '/workspace/ssd1/users/jhbyun/datasets/aistplusplus/model_seperate_features/'
    os.makedirs(save_dir, exist_ok = True)
    cache_audio_features(audio_names, save_dir, src_dir, audio_dir)