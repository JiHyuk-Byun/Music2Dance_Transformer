"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layer.layer_norm import LayerNorm
from models.layer.multi_head_attention import MultiHeadAttention
from models.layer.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, ffn_dim, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # Self Attention
        self.self_attention = MultiHeadAttention(hid_dim=hid_dim, n_head=n_head)
        self.norm1 = LayerNorm(hid_dim=hid_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)
        # Cross Attention
        self.enc_dec_attention = MultiHeadAttention(hid_dim=hid_dim, n_head=n_head)
        self.norm2 = LayerNorm(hid_dim=hid_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)
        # FFN
        self.ffn = PositionwiseFeedForward(hid_dim=hid_dim, hidden=ffn_dim, drop_prob=drop_prob) # Single hidden layer
        self.norm3 = LayerNorm(hid_dim=hid_dim)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, query, context):

        # Masked SA
        # 1. compute self attention with mask
        _x = query
        x = self.self_attention(q=query, k=query, v=query)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Cross Attention
        if context is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=context, v=context)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # FFN
        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
