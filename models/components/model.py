import torch
from torch import nn
import numpy as np

from utils.smoothe import smooth_pose
import utils.rotation_conversions as geometry 
from models.components.decoder import TransformerDecoder
from models.components.encoder import TransformerEncoder

class M2D(nn.Module):
    def __init__(self, 
    audio_channel=441,
    noise_size=256,
    dim=512, 
    depth=2, 
    heads=8, 
    mlp_dim=2048,
    music_length=240, 
    seed_m_length=120, 
    predict_length=20,
    smpl=None, 
    rot_6d=True,
    reconstruction=True,
    cross_attn=False,
    mint=False,
    device=None):

        super().__init__()
        self.reconstruction = reconstruction
        self.reconstruction_net = nn.Sequential(
            nn.Linear(10, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU()
        )

        self.music_length = music_length
        self.seed_m_length = seed_m_length
        self.predict_length = predict_length

        self.rot_6d = rot_6d

        self.mapping = MappingNet(noise_size, dim)

        self.emb_a = nn.Linear(audio_channel, dim)
        self.emb_a_mint = nn.Linear(audio_channel, dim)

        self.pos_encoder = PositionalEncoding(d_model=dim, max_len=seed_m_length + music_length)
        
        self.cross_attn = cross_attn

        self.mint = mint

        if rot_6d: #input이 6d인 경우.
            self.emb_m = nn.Linear(24 * 6 + 3, dim)
            self.emb_l = nn.Linear(dim, 24 * 6 + 3)
            self.emb_m_mint = nn.Linear(24 * 6 + 3, dim)
        else:
            self.emb_m = nn.Linear(24 * 3 + 3, dim)
            self.emb_l = nn.Linear(dim, 24 * 3 + 3)
            self.emb_m_mint = nn.Linear(24 * 3 + 3, dim)
        
        if self.cross_attn:
            self.tr_block = TransformerDecoder(
                    in_len=music_length + seed_m_length, 
                    hid_dim=dim,       
                    ffn_dim=mlp_dim,        
                    n_head=heads,         
                    n_layers=depth,       
                    drop_prob=0.1,      
                    device=device

            )

        elif self.mint:
            self.tr_block_a = TransformerEncoder(
                    in_len=music_length, 
                    hid_dim=dim,       
                    ffn_dim=mlp_dim,        
                    n_head=10,         
                    n_layers=2,       
                    drop_prob=0.1,      
                    device=device
            )

            self.tr_block_m = TransformerEncoder(
                    in_len=seed_m_length, 
                    hid_dim=dim,       
                    ffn_dim=mlp_dim,        
                    n_head=10,         
                    n_layers=2,       
                    drop_prob=0.1,      
                    device=device
            )

            self.tr_block = TransformerEncoder(
                    in_len=music_length + seed_m_length, 
                    hid_dim=dim,       
                    ffn_dim=mlp_dim,        
                    n_head=heads,         
                    n_layers=depth,       
                    drop_prob=0.1,      
                    device=device
            )

        else:
            self.emb_genre = nn.Linear(10, dim)
            self.tr_block = TransformerEncoder(
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
        
        if self.cross_attn:
            a = self.pos_encoder(self.emb_a(audio))
            # 'a': [batch, music_length, dim]
            m = self.pos_encoder(self.emb_m(motion))
            # 'm': [batch, seed_m_length, dim]

            x = torch.cat([m, a], dim=1) # audio|motion을 query로 사용
            # 'x': [batch, music_length+seed_m_length, dim]
            if self.reconstruction:
                s = nn.functional.one_hot(genre, 10).to(torch.float32)
                s = self.reconstruction_net(s)[:, None]
            else:
                s = self.mapping(noise, genre)[:, None] # noise를 주어진 genre에 맞는 network를 거쳐 genre feature를 구함.
                # 's': [batch, 1, dim]

            x = self.tr_block(x, s)

        elif self.mint:
            a = self.pos_encoder(self.emb_a_mint(audio))
            # 'a': [batch, music_length, dim]

            m = self.pos_encoder(self.emb_m_mint(motion))
            # 'm': [batch, seed_m_length, dim]

            a = self.tr_block_a(a)
            # 'a': [batch, music_length, dim]

            m = self.tr_block_m(m)
            # 'm': [batch, seed_m_length, dim]

            x = torch.cat([m, a], dim=1)
            # 'x': [batch, music_length+seed_m_length, dim]

            x = self.tr_block(x)
            # 'x': [batch, music_length+seed_m_length+1, dim]

        else: # reconstruction + use encoder
            # g = nn.functional.one_hot(genre, 10).to(torch.float32)
            # g = self.emb_genre(g)[:, None]
            # 'g': [batch, dim]

            a = self.pos_encoder(self.emb_a(audio))
            # 'a': [batch, music_length, dim]

            # a_genre = self.pos_encoder(torch.cat([g,a], dim=1))
            # 'a_genre': [batch, music_length+1, dim]

            m = self.pos_encoder(self.emb_m(motion))
            # 'm': [batch, seed_m_length, dim]

            x = torch.cat([m, a], dim=1) # audio|motion을 input으로
            # 'x': [batch, music_length+seed_m_length, dim]

            x = self.tr_block(x)
            # 'x': [batch, music_length+seed_m_length, dim]
    
        # Head
        x = self.emb_l(x)[:, :self.predict_length] # slice only prediction_length
        # 'x': [batch, music_length+seed_m_length+1, 24*6+3]

        output = x
        return output

    def inference(self, audio, motion, noise, genre):
        T = audio.shape[1] # time
        b, s, _ = motion.shape

        new_motion = motion # seed motion
        # [batch, seed_m_length, 24*3+3]
        for idx in range(0, T - self.music_length + 1, self.predict_length):
            audio_ = audio[:, idx:idx + self.music_length]

            motion_ = new_motion[:, -self.seed_m_length:]
            # [batch, seed_m_length, 24*3+3]
            motion_ = self.forward(audio_, motion_, noise, genre)
            # [batch, predict_length, 24*6+3]
            if self.rot_6d:
                motion_ = geometry.rot6dTOmat(motion_) 
            # [batch, predict_length, 24*3+3]
            b, p, _ = motion_.shape

            # # Smooth motion
            # for idx, motion in enumerate(motion_):
            #     pose, trans = motion[:, :-3].view(-1, 24, 3), motion[:, -3:]
            #     smth_poses = torch.from_numpy(smooth_pose(pose.cpu().numpy()))
            #     smth_trans = torch.from_numpy(smooth_pose(trans.cpu().numpy()))

            #     motion_[idx] = torch.cat([smth_poses.view(p, -1), smth_trans.view(p, -1)], dim=1)

            new_motion = torch.cat([new_motion, motion_], dim=1)
            
        return new_motion

class MappingNet(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()

        self.eps = 1e-8
        # Initialize learnable distribution for each genres
        self.mu_list = []
        self.sigma_list = []
        for _ in range(10):
            self.mu_list.append(nn.Parameter(torch.randn(in_dim)))
            self.sigma_list.append(nn.Parameter(torch.randn(in_dim)))

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

    def forward(self, noise, genre_batch):
        
        mu_batch = []
        sigma_batch = []
        for i, genre in enumerate(genre_batch):
            mu = self.mu_list[genre]
            sigma = self.sigma_list[genre]
            mu_batch.append(mu)
            sigma_batch.append(sigma)
    
        mu = torch.stack(mu_batch).to(genre.device)
        sigma = torch.stack(sigma_batch).to(genre.device)

        # "mu", "sigma": [batch, noise_dim]

        noise_genre = torch.normal(mu, torch.abs(sigma) + self.eps)
        s = self.shared(noise_genre) # noise input을 MLP로 어떤 shared 차원으로 이동.
        # 's' : [batch, dim]
        out = []
        
        # genre is list and network selection can not be batched. therefore we need to pass genre per 1 batch
        for i, genre in enumerate(genre_batch):
            net = self.unshared[genre]

            g_feature = net(s[i].unsqueeze(0))
            
            out.append(g_feature)

        # Concat outputs to reassemble as batches
        out = torch.cat(out, dim=0)
        # out: [batch, hid_dim]
        return out
        
        # sList = []
        # for unshare in self.unshared: # 해당 feature를 각 10개의 genre별로 대응시킴
        #     sList.append(unshare(s))

        # s = torch.stack(sList, dim=1)
        # idx = torch.LongTensor(range(len(genre))).to(genre.device)
        # return s[idx, genre]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model

        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

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
