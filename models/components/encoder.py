"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.components.blocks.encoder_layer import EncoderLayer
from models.components.embedding.transformer_embedding import TransformerEmbedding
from models.components.embedding.token_embedding import TokenEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self,
    in_len,        # max sequence length of input
    hid_dim,        # d_model
    ffn_dim,        # ffn_hidden(Feed Forward hidden dimension)
    n_head,         # MHA
    n_layers,       # number of attention layers
    drop_prob,      # drop out
    device):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hid_dim=hid_dim,
                                                  ffn_dim=ffn_dim,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])


    def forward(self, query):
        _, length, dim = query.shape

        for layer in self.layers:
            query = layer(query) # norm(MSA)-> norm(MHA) -> norm(FFN) ->

        # pass to LM head
        output = query
        return output
