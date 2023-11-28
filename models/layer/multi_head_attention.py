"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, hid_dim, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention() # attention per query
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.w_concat = nn.Linear(hid_dim, hid_dim)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask) # return: v, attention

        # 4. concat multi heads and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    # Split hidden dimension by number of heads
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, hid_dim]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, hid_dim = tensor.size()

        tensor = tensor.view(batch_size, length, self.n_head, -1).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    # concat multi-heads
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, hid_dim]
        """
        batch_size, head, length, d_tensor = tensor.size()

        # Contiguous(): to prevent RuntimeError: input is not contiguous.
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, -1) 
        return tensor
