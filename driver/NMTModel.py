# -*- coding: utf-8 -*-

from module.Embedding import *
from module.Encoder import EncoderRNN
from module.Decoder import AttnDecoderRNN
from driver.Utils import *
import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)

class NeuralMT(nn.Module):
    def __init__(self, config, src_vocab_size, src_rel_size, tgt_vocab_size, padding_idx):
        super(NeuralMT, self).__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.embedding_encoder = Embedding(src_vocab_size, config.embed_size, padding_idx=padding_idx)
        self.rel_embedding_encoder = Embedding(src_rel_size, config.rel_embed_size, padding_idx=padding_idx)
        self.embedding_decoder = Embedding(tgt_vocab_size, config.embed_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(config.dropout_hidden)

        self.encoder = EncoderRNN(config.embed_size + config.rel_embed_size,
                                  config.hidden_size,
                                  config.num_layers,
                                  config.dropout_hidden)

        self.decoder = AttnDecoderRNN(config.embed_size,
                                      2*config.hidden_size,
                                      config.num_layers,
                                      config.dropout_hidden)

        self.predictor = nn.Linear(2*config.hidden_size, tgt_vocab_size, bias=False)

        self.output = nn.LogSoftmax(dim=-1)

        weight = torch.ones(tgt_vocab_size)
        weight[padding_idx] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)

        for p in self.parameters():
            nn.init.uniform(p.data, -config.param_init, config.param_init)


    def encode(self, words, rels, heads, lengths=None):
        word_emb = self.embedding_encoder(words)
        rel_emb = self.rel_embedding_encoder(rels)
        encoder_input = torch.cat([word_emb, rel_emb], 2)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, heads, lengths, None)

        return encoder_outputs, encoder_hidden

    def decode(self, input, context, state):
        emb = self.embedding_decoder(input)
        emb = self.dropout(emb)
        decoder_outputs, decoder_hiddens = self.decoder(emb, context, state)
        return decoder_outputs, decoder_hiddens

    def forward(self, words, rels, heads, tgt_words, src_lengths):
        encoder_outputs, encoder_hidden = self.encode(words, rels, heads, src_lengths)
        decoder_init_hidden = self.decoder.init_decoder_state(encoder_hidden)
        decoder_outputs, decoder_hiddens = self.decode(tgt_words[:-1], \
                                     encoder_outputs, decoder_init_hidden)

        return decoder_outputs

    def analyze(self, words, rels, heads, tgt_words, src_lengths):
        encoder_outputs, encoder_hidden = self.encode(words, rels, heads, src_lengths)
        decoder_init_hidden = self.decoder.init_decoder_state(encoder_hidden)
        emb = self.embedding_decoder(tgt_words[:-1])
        emb = self.dropout(emb)
        decoder_outputs, decoder_hiddens, attn_scores = self.decoder.analyze(emb, \
                                     encoder_outputs, decoder_init_hidden)

        attn_scores = torch.squeeze(attn_scores, dim=1)
        attn_scores = attn_scores.data.cpu().numpy()
        return decoder_outputs, attn_scores


    def compute_loss(self, tgt_inputs, logits):
        output = logits.view(-1, logits.size(2))
        scores = self.predictor(output)
        scores = self.output(scores)
        target = tgt_inputs[1:].view(-1)
        loss = self.criterion(scores, target)
        loss_data = loss.data.clone()
        score_data = scores.data.clone()
        target_data = target.data.clone()
        stats = self.stats(loss_data[0], score_data, target_data)
        return loss, stats


    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.
        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_words = non_padding.sum()
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return Statistics(loss, num_words, num_correct)


