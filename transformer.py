import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

    
class ConvTSEmbedding(nn.Module):
    '''
    Causal convolutional embedding for time series value.
    Convolutions are only applied to the left. (causal convolution)
    '''
    def __init__(self, embedding_dim, kernel_size=3, conv_depth=4, input_channel=1):
        super(ConvTSEmbedding, self).__init__()

        self.fc = nn.Linear(input_channel, embedding_dim)

        self.kernel_size = kernel_size
        self.conv_list = nn.ModuleList([nn.Conv1d(
            in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=self.kernel_size) for _ in range(conv_depth)])


    def forward(self, x): # (N, seq_len, input_channel)
        x = self.fc(x) # (N, seq_len, embedding_dim)

        x = x.permute(0, 2, 1) # (N, embedding_dim, seq_len)
        for conv in self.conv_list:
            x = F.pad(x, (self.kernel_size-1,0))
            x = conv(x)

        return x.permute(0, 2, 1) # (N, seq_len, embedding_dim)


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super(LearnedPositionEmbedding, self).__init__()

        pos_tensor = torch.arange(seq_len)
        self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

        self.register_buffer('pos_tensor', pos_tensor)

    def forward(self, x): # x.shape == (N, ...)
        pos_embedded = self.pos_embedding(self.pos_tensor) # pos_embedded.shape == (seq_len, embedding_dim)
        return pos_embedded.repeat(x.shape[0], 1, 1) # (N, seq_len, embedding_dikm)


class FixedPositionEmbedding(nn.Module):
    '''
    Fixed position embedding in "Attention is all you need".
    Code from "Informer".
    '''
    def __init__(self, seq_len, embedding_dim):
        super(FixedPositionEmbedding, self).__init__()

        pos_embedding = torch.zeros((seq_len, embedding_dim)).float()
        pos_embedding.requires_grad = False

        pos_tensor = torch.arange(seq_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_dim, 2).float()
                    * -(math.log(10000.0) / embedding_dim)).exp()

        pos_embedding[:, 0::2] = torch.sin(pos_tensor * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos_tensor * div_term)

        pos_embedding.unsqueeze_(0) # dimension for batch

        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x): # (N, ...)
        return self.pos_embedding.repeat(x.shape[0], 1, 1) # (N, seq_len, embedding_dim)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        if attn_mask is not None:
        	# 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=3, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention



class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim, ffn_dim, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.permute(0, 2, 1)
        output = self.w2(F.relu(self.w1(output)))
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        # add residual and norm layer
        # print('output:', output.shape)
        output = self.layer_norm(x + output)
        return output


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask




class EncoderLayer(nn.Module):
    """Encoder的一层"""

    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        #self.freqatt = FrequencyAttention(K = 4, dropout = dropout)
        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(n_feature, hid_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        # print('context:', context.shape)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class DecoderLayer(nn.Module):

    def __init__(self, n_feature, num_heads, hid_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_feature, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(n_feature, hid_dim, dropout)

    def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention
