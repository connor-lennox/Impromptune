from math import sqrt

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, key_dim=64, value_dim=64, n_heads=8, relative_cutoff=128):
        super().__init__()

        self.value_dim = value_dim
        # Relative cutoff in each direction
        self.relative_cutoff = relative_cutoff

        # Typical multi-head attention parameters
        self.w_q = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_k = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_v = nn.Parameter(torch.Tensor(n_heads, embed_dim, value_dim))
        self.w_o = nn.Parameter(torch.Tensor(n_heads))

        # Relative positional values for each element relative_cutoff out from center, plus one for center
        self.a_k = nn.Parameter(torch.Tensor(relative_cutoff*2+1, key_dim))
        self.a_v = nn.Parameter(torch.Tensor(relative_cutoff*2+1, value_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_q, a=sqrt(5))
        init.kaiming_uniform_(self.w_k, a=sqrt(5))
        init.kaiming_uniform_(self.w_v, a=sqrt(5))
        init.ones_(self.w_o)

        init.kaiming_uniform_(self.a_k, a=sqrt(5))
        init.kaiming_uniform_(self.a_v, a=sqrt(5))

    def forward(self, xs):
        """Sequence should come in as the format (batch, seq_length, embed_dim)"""

        sequence_length = xs.shape[1]

        # Standard calculation of queries, keys, and values for each element of the input sequence
        # Each has shape (batch, n_head, seq_len, result), where result is the query/key/value dimension
        queries = torch.einsum('bse,heq->bhsq', xs, self.w_q)
        keys = torch.einsum('bse,hek->bhsk', xs, self.w_k)
        values = torch.einsum('bse,hev->bhsv', xs, self.w_v)

        # queries * keys^T
        q_k = torch.einsum('bhiq,bhjq->bhij', queries, keys)

        # queries * relative key encoding
        q_rel = torch.einsum('bhiq,ijq->bhij',
                             queries, self._generate_relative_positional_embeddings(self.a_k, sequence_length))

        # sum the standard q*k component and the relative positional query component
        # division by sqrt(value_dim) for normalization
        e = (q_k + q_rel) / sqrt(self.value_dim)
        alphas = F.softmax(e, dim=3)

        # alphas * values (summed over j axis)
        # qkv shape = (batch, heads, seq_len, value_dim)
        qkv = torch.einsum('bhij,bhjv->bhiv', alphas, values)

        # alphas * relative value encoding (summed over j axis)
        v_rel = torch.einsum('bhij,ijv->bhiv',
                             alphas, self._generate_relative_positional_embeddings(self.a_v, sequence_length))

        # z of shape (batch, n_head, seq_len, value_dim)
        z = qkv + v_rel

        # Combine heads with scaling factors in w_o
        result = torch.einsum('bhsv,h->bsv', z, self.w_o)

        # Final output shape: (batch, seq_len, value_dim)
        return result

    def _generate_relative_positional_embeddings(self, embeddings_matrix, sequence_length):
        pos_matrix = torch.arange(0, sequence_length).repeat(sequence_length, 1)
        rel_matrix = pos_matrix - torch.transpose(pos_matrix, 0, 1)
        rel_matrix = torch.clamp(rel_matrix, -self.relative_cutoff, self.relative_cutoff)
        rel_matrix = rel_matrix + self.relative_cutoff
        return embeddings_matrix[rel_matrix]


class PredictiveRelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, key_dim=64, value_dim=64, n_heads=8, relative_cutoff=128):
        super().__init__()

        self.relative_cutoff = relative_cutoff
        self.value_dim = value_dim

        # Typical multi-head attention parameters
        self.w_q = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_k = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_v = nn.Parameter(torch.Tensor(n_heads, embed_dim, value_dim))
        self.w_o = nn.Parameter(torch.Tensor(n_heads))

        # Relative positional values go only backwards now
        self.a_k = nn.Parameter(torch.Tensor(relative_cutoff+1, key_dim))
        self.a_v = nn.Parameter(torch.Tensor(relative_cutoff+1, value_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_q, a=sqrt(5))
        init.kaiming_uniform_(self.w_k, a=sqrt(5))
        init.kaiming_uniform_(self.w_v, a=sqrt(5))
        init.ones_(self.w_o)

        init.kaiming_uniform_(self.a_k, a=sqrt(5))
        init.kaiming_uniform_(self.a_v, a=sqrt(5))

    def forward(self, xs):
        """xs of shape (batch, seq_len, embed_dim)"""

        seq_len = xs.shape[1]

        # Only the last element of the sequence will be used to generate a query:
        query = torch.einsum('be,heq->bhq', xs[:, -1, :], self.w_q)

        # Keys and values still calculated for every element:
        keys = torch.einsum('bse,hek->bhsk', xs, self.w_k)
        values = torch.einsum('bse,hev->bhsv', xs, self.w_v)

        # Query x keys
        q_k = torch.einsum('bhq,bhsq->bhs', query, keys)

        # Query x rel. key encoding
        q_rel = torch.einsum('bhq,sq->bhs',
                             query, self._generate_relative_positional_embeddings(self.a_k, seq_len))

        # Shape of e: (batch, head, seq_len)
        e = (q_k + q_rel) / sqrt(self.value_dim)
        alphas = F.softmax(e, dim=2)

        # Application of alphas to values, summed over j axis
        qkv = torch.einsum('bhs,bhsv->bhv', alphas, values)

        # Alphas x rel. value encoding
        v_rel = torch.einsum('bhs,sv->bhv',
                             alphas, self._generate_relative_positional_embeddings(self.a_v, seq_len))

        # z of shape (batch, n_heads, value_dim)
        z = qkv + v_rel

        # Combine heads with w_o weight vector
        result = torch.einsum('bhv,h->bv', z, self.w_o)

        # Final output shape: (batch, value_dim)
        return result

    def _generate_relative_positional_embeddings(self, embeddings_matrix, seq_len):
        """Generates in O(LD) memory space, since only the last element of the sequence is being queried """
        pos_vec = torch.flip(torch.arange(0, seq_len), [0])
        pos_vec = torch.clamp(pos_vec, 0, self.relative_cutoff)
        return embeddings_matrix[pos_vec]


if __name__ == '__main__':
    # parameter of (embed)
    test_att = PredictiveRelativeMultiheadAttention(128, value_dim=128)
    # input of shape (batch,seq,embed)
    test_input = torch.randn((256, 1000, 128))
    test_result = test_att(test_input)
    # shape should be (batch,value_dim), as only one output is generated per input sequence
    print(test_result.shape)
