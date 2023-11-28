"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding
from embedding.token_embedding import TokenEmbedding

class TransformerDecoder(nn.Module):
    def __init__(self,
    in_len,        # max sequence length of input
    hid_dim,        # d_model
    ffn_dim,        # ffn_hidden(Feed Forward hidden dimension)
    n_head,         # MHA
    n_layers,       # number of attention layers
    drop_prob,      # drop out
    device):
        super().__init__()

        # Embedding
        self.pos_emb = nn.Parameter(torch.randn(hid_dim, max_len))

        self.layers = nn.ModuleList([DecoderLayer(hid_dim=hid_dim,
                                                  ffn_dim=ffn_dim,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # head
        self.linear = nn.Linear(hid_dim, dance_channel)

    def forward(self, query, value, query_mask, value_mask):
        _, _, h, w = query.shape

        for layer in self.layers:
            query += self.pos_emb[:h, :w] # Learnable PE
            query = layer(query, value, query_mask, value_mask) # norm(MSA)-> norm(MHA) -> norm(FFN) ->

        # pass to LM head
        output = query
        return output
