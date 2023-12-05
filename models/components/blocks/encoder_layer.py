"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.components.layer.layer_norm import LayerNorm
from models.components.layer.multi_head_attention import MultiHeadAttention
from models.components.layer.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, ffn_dim, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # Self Attention
        self.self_attention = MultiHeadAttention(hid_dim=hid_dim, n_head=n_head)
        self.norm1 = LayerNorm(hid_dim=hid_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        # FFN
        self.ffn = PositionwiseFeedForward(hid_dim=hid_dim, hidden=ffn_dim, drop_prob=drop_prob) # Single hidden layer
        self.norm2 = LayerNorm(hid_dim=hid_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, query):

        # Masked SA
        # 1. compute self attention with mask

        _x = query
        x = self.self_attention(q=query, k=query, v=query)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # FFN
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
