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
        # Here q is the "query/key" dimension, which are guaranteed to be the same length
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


class EfficientRelativeMultiheadAttention(nn.Module):
    """Improvements on Relative Multihead Attention as described in Music Transformer (Huang et. al., 2018)"""
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
        """xs of shape (batch, seq_len, embed_dim)"""
        seq_len = xs.shape[1]

        # Standard calculation of queries, keys, and values for each element of the input sequence
        # Each has shape (batch, n_head, seq_len, result), where result is the query/key/value dimension
        queries = torch.einsum('bse,heq->bhsq', xs, self.w_q)
        keys = torch.einsum('bse,hek->bhsk', xs, self.w_k)
        values = torch.einsum('bse,hev->bhsv', xs, self.w_v)

        # queries * keys^T
        # Here q is the "query/key" dimension, which are guaranteed to be the same length
        q_k = torch.einsum('bhiq,bhjq->bhij', queries, keys)

        # queries x relative positional encodings
        # using "skewing" technique
        # This matrix has shape (batch, n_heads, seq_len, seq_len*2+1), and the elements are not lined up properly.
        q_rel = torch.einsum('bhiq,rq->bhir',
                             queries, self._generate_relative_positional_embeddings(self.a_k, seq_len))
        q_rel = self._skew_matrix(q_rel)

        # sum the standard q*k component and the relative positional query component
        # division by sqrt(value_dim) for normalization
        e = (q_k + q_rel) / sqrt(self.value_dim)
        alphas = F.softmax(e, dim=3)

        # alphas * values (summed over j axis)
        # qkv shape = (batch, heads, seq_len, value_dim)
        qkv = torch.einsum('bhij,bhjv->bhiv', alphas, values)

        # Temporarily dropping relative value calculation (is it required? Maybe not.)
        # v_rel = torch.einsum('bhij,rv->bhir',
        #                      alphas, self._generate_relative_positional_embeddings(self.a_v, seq_len))
        #
        # v_rel = self._skew_matrix(v_rel)

        # z of shape (batch, n_head, seq_len, value_dim)
        z = qkv  # + v_rel

        # Combine heads with scaling factors in w_o
        result = torch.einsum('bhsv,h->bsv', z, self.w_o)

        # Final output shape: (batch, seq_len, value_dim)
        return result

    def _generate_relative_positional_embeddings(self, embedding_matrix, seq_len):
        pos_vec = torch.arange(-seq_len+1, seq_len)
        pos_vec = torch.clamp(pos_vec, -self.relative_cutoff, self.relative_cutoff)
        pos_vec += self.relative_cutoff
        return embedding_matrix[pos_vec]

    @staticmethod
    def _skew_matrix(matrix):
        seq_len = matrix.shape[2]
        matrix = F.pad(matrix, [0, 1])          # Pad one column on right side of data
        matrix = torch.flatten(matrix, 2, 3)    # Flatten last two dimensions
        matrix = F.pad(matrix, [0, seq_len-1])  # Pad an additional seq_len-1 elements
        matrix = torch.reshape(matrix, (matrix.shape[0], matrix.shape[1], seq_len+1, 2*seq_len-1))  # Reshape matrix
        matrix = matrix[:, :, :seq_len, matrix.shape[3]-seq_len:]   # Slice to retain first seq_len rows and last seq_len columns
        return matrix


class LocalRelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, key_dim=64, value_dim=64, n_heads=8, look_back=128, look_forward=0):
        super().__init__()

        self.value_dim = value_dim
        self.look_back = look_back
        self.look_forward = look_forward

        # Standard attention parameters
        self.w_q = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_k = nn.Parameter(torch.Tensor(n_heads, embed_dim, key_dim))
        self.w_v = nn.Parameter(torch.Tensor(n_heads, embed_dim, value_dim))
        self.w_o = nn.Parameter(torch.Tensor(n_heads))

        # Relative position parameter (applied only to queries) based on lookback/forward
        self.a_k = nn.Parameter(torch.Tensor(look_back+1+look_forward, key_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_q, a=sqrt(5))
        init.kaiming_uniform_(self.w_k, a=sqrt(5))
        init.kaiming_uniform_(self.w_v, a=sqrt(5))
        init.ones_(self.w_o)
        init.kaiming_uniform_(self.a_k, a=sqrt(5))

    def forward(self, xs):
        """xs of shape (batch, seq_len, embed_dim)"""
        seq_len = xs.shape[1]
        local_mask = self._generate_local_mask(seq_len, xs.device)

        # Standard calculation of queries, keys, and values for each element of the input sequence
        # Each has shape (batch, n_head, seq_len, result), where result is the query/key/value dimension
        queries = torch.einsum('bse,heq->bhsq', xs, self.w_q)
        keys = torch.einsum('bse,hek->bhsk', xs, self.w_k)
        values = torch.einsum('bse,hev->bhsv', xs, self.w_v)

        # queries * keys^T
        # Here q is the "query/key" dimension, which are guaranteed to be the same length
        q_k = torch.einsum('bhiq,bhjq->bhij', queries, keys)

        # queries x relative positional encodings
        # using "skewing" technique
        # This matrix has shape (batch, n_heads, seq_len, seq_len*2+1), and the elements are not lined up properly.
        q_rel = torch.einsum('bhiq,rq->bhir',
                             queries, self._generate_relative_positional_embeddings(self.a_k, seq_len))
        q_rel = self._skew_matrix(q_rel)

        # sum the standard q*k component and the relative positional query component
        # division by sqrt(value_dim) for normalization
        e = (q_k + q_rel) / sqrt(self.value_dim)
        alphas = F.softmax(e, dim=3)

        # Mask out "non-local" values for element compatibility
        # Logically, you are completely incompatible with elements outside of your neighborhood
        alphas = torch.einsum('bhij,ij->bhij', alphas, local_mask)

        # alphas * values (summed over j axis)
        # qkv shape = (batch, heads, seq_len, value_dim)
        qkv = torch.einsum('bhij,bhjv->bhiv', alphas, values)

        # Combine heads with scaling factors in w_o
        result = torch.einsum('bhsv,h->bsv', qkv, self.w_o)

        # Final output shape: (batch, seq_len, value_dim)
        return result

    def _generate_local_mask(self, seq_len, device):
        mask = torch.tensor([0] + [1] * (self.look_back+1+self.look_forward) + [0] * (seq_len-1-self.look_forward)).to(device)
        mask = mask.repeat(seq_len)
        if self.look_back >= seq_len:
            mask = F.pad(mask, (0, self.look_back-seq_len+1))
            mask = mask[self.look_back+1:]
        else:
            right_end = None if abs(seq_len-self.look_back) == 1 else -(seq_len-self.look_back-1)
            mask = mask[self.look_back+1:right_end]
        mask = mask.reshape(seq_len, -1)
        mask = mask[:seq_len, :seq_len]
        return mask

    def _generate_relative_positional_embeddings(self, embedding_matrix, seq_len):
        pos_vec = torch.arange(-seq_len+1, seq_len)
        pos_vec = torch.clamp(pos_vec, -self.look_back, self.look_forward)
        pos_vec += self.look_back
        return embedding_matrix[pos_vec]

    @staticmethod
    def _skew_matrix(matrix):
        seq_len = matrix.shape[2]
        matrix = F.pad(matrix, [0, 1])          # Pad one column on right side of data
        matrix = torch.flatten(matrix, 2, 3)    # Flatten last two dimensions
        matrix = F.pad(matrix, [0, seq_len-1])  # Pad an additional seq_len-1 elements
        matrix = torch.reshape(matrix, (matrix.shape[0], matrix.shape[1], seq_len+1, 2*seq_len-1))  # Reshape matrix
        matrix = matrix[:, :, :seq_len, matrix.shape[3]-seq_len:]   # Slice to retain first seq_len rows and last seq_len columns
        return matrix


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
        z = qkv #+ v_rel

        # Combine heads with w_o weight vector
        result = torch.einsum('bhv,h->bv', z, self.w_o)

        # Final output shape: (batch, value_dim)
        return result

    def _generate_relative_positional_embeddings(self, embeddings_matrix, seq_len):
        """Generates in O(LD) memory space, since only the last element of the sequence is being queried """
        pos_vec = torch.flip(torch.arange(0, seq_len), [0])
        pos_vec = torch.clamp(pos_vec, 0, self.relative_cutoff)
        return embeddings_matrix[pos_vec]


class InformedPredictiveAttention(nn.Module):
    def __init__(self, embed_dim, key_dim=64, value_dim=64, n_heads=8, relative_cutoff=128):
        super().__init__()

        self.relative_cutoff = relative_cutoff

        self.rel_attn = EfficientRelativeMultiheadAttention(embed_dim, key_dim, value_dim, n_heads, relative_cutoff)

        # Relative parameter for informed weighting
        self.a_w = nn.Parameter(torch.Tensor(relative_cutoff+1))

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.a_w)

    def forward(self, xs):
        # Input xs: (batch, seq_length, embed_dim)
        # Run elements through relative attention
        xs = self.rel_attn(xs)      # (batch, seq_len, value_dim)

        seq_len = xs.shape[1]

        # Combine all values by learned weights
        # TODO: Check if the softmax is really necessary here
        w = F.softmax(self._generate_sequence_weights(self.a_w, seq_len), dim=0)
        result = torch.einsum("bsv,s->bv", xs, w)

        # Final output shape: (batch, value_dim)
        return result

    def _generate_sequence_weights(self, weights_vector, seq_len):
        pos_vec = torch.flip(torch.arange(0, seq_len), [0])
        pos_vec = torch.clamp(pos_vec, 0, self.relative_cutoff)
        return weights_vector[pos_vec]


if __name__ == '__main__':
    # # parameter of (embed)
    # test_att = PredictiveRelativeMultiheadAttention(128, value_dim=128)
    # # input of shape (batch,seq,embed)
    # test_input = torch.randn((256, 1000, 128))
    # test_result = test_att(test_input)
    # # shape should be (batch,value_dim), as only one output is generated per input sequence
    # print(test_result.shape)
    # test_eff_att = EfficientRelativeMultiheadAttention(3, key_dim=4, value_dim=4, n_heads=1, relative_cutoff=1)
    # test_att = RelativeMultiheadAttention(3, key_dim=4, value_dim=4, n_heads=1, relative_cutoff=1)
    # test_att.load_state_dict(test_eff_att.state_dict())
    # test_input = torch.randn((1, 3, 3))
    # # These results won't be the same since the efficient attention does not do relative positional values,
    # # but calculating both provides an entry to the function so that a breakpoint can look at the intermediary
    # # relative query position results.
    # test_result = test_att(test_input)
    # test_eff_result = test_eff_att(test_input)
    # print(test_result.shape)

    # test_local_attn = LocalRelativeMultiheadAttention(3, key_dim=4, value_dim=4, n_heads=8, look_back=10, look_forward=10)
    test_input = torch.randn(8, 9, 3)
    # test_output = test_local_attn(test_input)
    # print(test_output.shape)
    test_pred = InformedPredictiveAttention(3, key_dim=4, value_dim=4, n_heads=8, relative_cutoff=4)
    test_output = test_pred(test_input)
    print(test_output.shape)
