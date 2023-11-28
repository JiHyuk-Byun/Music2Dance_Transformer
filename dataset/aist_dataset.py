from torch.utils.data import Dataset
from data_utils import * # 동일폴더


class AISTPPDataset(Dataset):
    '''
    dance_s:  torch.Size([2, 120, 75]) [batch, seed_m_length, pose&trans]
    dance_f:  torch.Size([2, 20, 75])  [batch, predict_length, pose&trans]
    music:  torch.Size([2, 240, 441])  [batch, music_length, feature size]
    dance_id:  torch.Size([2])         [batch]
    '''
    def __init__(self, src_path, music_length, seed_m_length, predict_length):

        self.music_length = music_length
        self.seed_m_length = seed_m_length
        self.predict_length = predict_length
        # self.sample_length = seed_length + predict_length
        self.dance_dict, self.dance_list = processing_dance_list(src_path) # pkl => dance feature upload
        self.music_dict = processing_music_list(src_path) # npy-> music feature dictionary upload

    def __len__(self):
        return len(self.dance_list)

    def __getitem__(self, idx):
        motion_keys = self.dance_list[idx]
        keys = motion_keys.split('_')
        dance_genre = keys[0]
        music_name = keys[4]
        
        # get features of given idx
        pose, trans = self.dance_dict[motion_keys]
        music = self.music_dict[music_name].type(torch.float32)

        dance_id = torch.from_numpy(np.array(dance_genre_dict[dance_genre])).long() # dance genre's embedding(0~9)
        print(dance_id)
        sample_idx = get_feature_sample(pose.shape[0], self.music_length) # 시작 index sampling

        pose_s, pose_f = pose[sample_idx : sample_idx + self.seed_m_length],\
                         pose[sample_idx + self.seed_m_length : sample_idx + self.seed_m_length + self.predict_length]
        trans_s, trans_f = trans[sample_idx : sample_idx + self.seed_m_length], \
                           trans[sample_idx + self.seed_m_length : sample_idx + self.seed_m_length + self.predict_length]

        dance_s = torch.cat([pose_s, trans_s], dim=1)
        dance_f = torch.cat([pose_f, trans_f], dim=1)
        music = music[sample_idx:sample_idx+self.music_length]

        # '_s': 시작 frame, '_f': 끝 frame
        return dance_s, dance_f, music, dance_id


if __name__ == '__main__':
    from torch.utils import data
    import numpy as np
    data_path = "/workspace/ssd1/users/jhbyun/datasets/aistplusplus" # src_dir
    dataset = AISTPPDataset(data_path, 240, 120, 20)
    dataset[0]

    loader = iter(data.DataLoader(
        dataset,
        batch_size=2,
        sampler=data.RandomSampler(dataset),
        drop_last=False,
        num_workers=16
    ))

    for idx in range(len(loader)):
        try:
            dance_s, dance_f, music, dance_id = next(loader)
            print("dance_s: ", dance_s.size()) 
            print("dance_f: ", dance_f.size())
            print("music: ", music.size())
            print("dance_id: ", dance_id.size())
        except:
            pass