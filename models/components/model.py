import torch
from torch import nn

import utils.rotation_conversions as geometry 
from models.components.decoder import TransformerDecoder

class MappingNet(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        self.unshared = nn.ModuleList()        
        for _ in range(10):
            self.unshared.append(nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, dim)
            ))

    def forward(self, x, genre_batch):


        s = self.shared(x) # noise input을 MLP로 어떤 shared 차원으로 이동.
        # 's' : [batch, dim]
        unshared = self
        out = []
        
        # genre is list and network selection can not be batched. therefore we need to pass genre per 1 batch
        for i, genre in enumerate(genre_batch):
            net = self.unshared[genre]

            g_feature = net(s[i].unsqueeze(0))
            
            out.append(g_feature)

        # Concat outputs to reassemble as batches
        out = torch.cat(out, dim=0)

        return out
        
        # sList = []
        # for unshare in self.unshared: # 해당 feature를 각 10개의 genre별로 대응시킴
        #     sList.append(unshare(s))

        # s = torch.stack(sList, dim=1)
        # idx = torch.LongTensor(range(len(genre))).to(genre.device)
        # return s[idx, genre]

class M2D(nn.Module):
    def __init__(self, 
    audio_channel=441,
    noise_size=256,
    dim=512, 
    depth=4, 
    heads=8, 
    mlp_dim=2048,
    music_length=240, 
    seed_m_length=120, 
    predict_length=20,
    smpl=None, 
    rot_6d=True,
    device=None):

        super().__init__()
        self.music_length = music_length
        self.seed_m_length = seed_m_length
        self.predict_length = predict_length

        self.rot_6d = rot_6d

        self.mapping = MappingNet(noise_size, dim)

        self.mlp_a = nn.Linear(audio_channel, dim)

        if rot_6d: #input이 6d인 경우.
            self.mlp_m = nn.Linear(24 * 6 + 3, dim)
            self.mlp_l = nn.Linear(dim, 24 * 6 + 3)
        else:
            self.mlp_m = nn.Linear(24 * 3 + 3, dim)
            self.mlp_l = nn.Linear(dim, 24 * 3 + 3)

        self.tr_block = TransformerDecoder(
                in_len=music_length + seed_m_length, 
                hid_dim=dim,       
                ffn_dim=mlp_dim,        
                n_head=heads,         
                n_layers=depth,       
                drop_prob=0.1,      
                device=device

        )

    def forward(self, audio, motion, noise, genre):
        if self.rot_6d:
            motion = geometry.matTOrot6d(motion)

        a = self.mlp_a(audio)
        # 'a': [batch, music_length, dim]
        m = self.mlp_m(motion)
        # 'm': [batch, seed_m_length, dim]

        x = torch.cat([m, a], dim=1) # audio|motion을 query로 사용
        # 'x': [batch, music_length+seed_m_length, dim]

        s = self.mapping(noise, genre)[:, None] # noise를 주어진 genre에 맞는 network를 거쳐 genre feature를 구함.
        # 's': [batch, 1, dim]

        x = self.tr_block(x, s)
        # 'x': [batch, music_length+seed_m_length, dim]


        # Head
        x = self.mlp_l(x)[:, : self.predict_length] # slice only prediction_length

        return x

    def inference(self, audio, motion, noise, genre):
        T = audio.shape[1] # time

        new_motion = motion
        for idx in range(0, T - self.music_length + 1, self.predict_length):
            audio_ = audio[:, idx:idx + self.music_length]

            motion_ = new_motion[:, -self.seed_m_length:]
            motion_ = self(audio_, motion_, noise, genre)

            if self.rot_6d:
                motion_ = geometry.rot6dTOmat(motion_)

            new_motion = torch.cat([new_motion, motion_], dim=1)
        return new_motion


if __name__ == '__main__':
    ####################################################
    print("[*] Start DanceGeneratorAMNG_D")
    model = M2D(predict_length=30)

    audio = torch.randn(2, 240, 441)
    audioF = torch.randn(2, 60 * 11 + 90, 441)

    noise = torch.randn(2, 256)

    motion = torch.randn(2, 120, 24 * 3 + 3)
    genre = torch.tensor([3, 8])

    pred_motion = model(audio, motion, noise, genre)

    # l_motion = l_motion[:, :, :-3]
    logit = D(audio, l_motion, genre)


    print("[*] Finish DanceGeneratorAMNG_D")
    print("*******************************")
