# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from module.Attention import GlobalAttention



class AttnDecoderRNN(nn.Module):
    """ The GlobalAttention-based RNN decoder. """
    def __init__(self, input_size, hidden_size, \
                num_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        # Basic attributes.
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers)

        self.attn = GlobalAttention(hidden_size)

        # mlp wants it with bias
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, context, state):
        rnn_outputs, hidden = self.rnn(input, state)

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # concatenate
        targetL, batch, dim = rnn_outputs.size()
        concat_c = torch.cat([rnn_outputs, attn_outputs], 2).view(targetL*batch, -1)
        outputs = self.linear_out(concat_c).view(targetL, batch, dim)

        outputs = self.dropout(self.tanh(outputs))    # (input_len, batch, d)

        return outputs, hidden

    def analyze(self, input, context, state):
        rnn_outputs, hidden = self.rnn(input, state)

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_outputs.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1)              # (contxt_len, batch, d)
        )

        # concatenate
        targetL, batch, dim = rnn_outputs.size()
        concat_c = torch.cat([rnn_outputs, attn_outputs], 2).view(targetL*batch, -1)
        outputs = self.linear_out(concat_c).view(targetL, batch, dim)

        outputs = self.dropout(self.tanh(outputs))    # (input_len, batch, d)

        return outputs, hidden, attn_scores

    def init_decoder_state(self, enc_hidden):
        return enc_hidden
