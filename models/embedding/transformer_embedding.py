from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, in_channel, hid_dim, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param in_channel: channel size of input
        :param hid_dim: dimensions of model
        """

        super(TransformerEmbedding, self).__init__()
        self.token_emb = TokenEmbedding(in_channel, hid_dim) # make dance and music dimension same
        self.pos_emb = PositionalEncoding(hid_dim, max_len, device) # PE only for dance
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
