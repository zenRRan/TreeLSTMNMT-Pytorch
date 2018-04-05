from driver.NMTModel import *
import torch.nn.functional as F
from torch.autograd import Variable
from data.DataLoader import *
from driver.Beam import *
from module.Tree import *


class NMTHelper(object):
    def __init__(self, model, src_vocab, tgt_vocab, config):
        self.model = model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config

    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        for idx in range(self.config.max_train_length):
            self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            idx = len(src_input)-1
            words, heads, rels = self.src_data_id(src_input)
            root, tree = creatTree(heads)
            if root.depth() > idx + 1:
                forms = [the_form + "_" + str(the_head) + "_" +  the_rel for the_form, the_head, the_rel in src_input]
                print("strange: " + '_'.join(forms))
            self.train_data[idx].append((words, rels, heads, self.tgt_data_id(tgt_input)))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        for idx in range(self.config.max_train_length):
            train_size = len(self.train_data[idx])
            batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_valid_data(self, src_inputs, tgt_inputs):
        self.valid_data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            words, heads, rels = self.src_data_id(src_input)
            self.valid_data.append((words, rels, heads, self.tgt_data_id(tgt_input)))
        self.valid_size = len(self.valid_data)

    def src_data_id(self, src_input):
        words, heads, rels = [self.src_vocab.ROOT], [-1], [self.src_vocab.ROOT]
        for unit in src_input:
            words.append(self.src_vocab.word2id(unit[0]))
            rels.append(self.src_vocab.rel2id(unit[2]))
            heads.append(unit[1])
        return words, heads, rels

    def tgt_data_id(self, tgt_input):
        result = self.tgt_vocab.word2id(tgt_input)
        return [self.tgt_vocab.BEGIN_SEQ] + result + [self.tgt_vocab.END_SEQ]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            words, heads, rels = self.src_data_id(src_input)
            forms = [the_form for the_form, the_head, the_rel in src_input]
            eval_data.append((words, rels, heads, forms))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        tgt_lengths = [len(batch[i][3]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        src_rels = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        tgt_words = Variable(torch.LongTensor(max_tgt_length, batch_size).zero_(), requires_grad=False)
        heads = []

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[index, b] = word
            for index, rel in enumerate(instance[1]):
                src_rels[index, b] = rel
            heads.append(instance[2])
            for index, word in enumerate(instance[3]):
                tgt_words[index, b] = word
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            src_rels = src_rels.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)

        return src_words, src_rels, heads, src_lengths, tgt_words

    def source_data_variable(self, batch):
        batch_size = len(batch)
        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        src_rels = Variable(torch.LongTensor(max_src_length, batch_size).zero_(), requires_grad=False)
        heads = []

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[index, b] = word
            for index, rel in enumerate(instance[1]):
                src_rels[index, b] = rel
            heads.append(instance[2])
            b += 1

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            src_rels = src_rels.cuda(self.device)

        return src_words, src_rels, heads, src_lengths

    def train_one_batch(self, batch):
        self.model.train()
        self.model.zero_grad()
        batch_size = len(batch)
        src_words, src_rels, heads, src_lengths, tgt_words = self.pair_data_variable(batch)
        decoder_outputs = self.model.forward(src_words, src_rels, heads, tgt_words, src_lengths)
        loss, stat = self.model.compute_loss(tgt_words, decoder_outputs)
        loss = loss / (batch_size * self.config.update_every)
        loss.backward()

        return stat

    def valid(self, global_step):
        valid_stat = Statistics()
        self.model.eval()
        for batch in create_batch_iter(self.valid_data, self.config.test_batch_size):
            src_words, src_rels, heads, src_lengths, tgt_words = self.pair_data_variable(batch)
            decoder_outputs = self.model.forward(src_words, src_rels, heads, tgt_words, src_lengths)
            loss, stat = self.model.compute_loss(tgt_words, decoder_outputs)
            valid_stat.update(stat)
        valid_stat.print_valid(global_step)
        return valid_stat

    def analyze(self):
        self.model.eval()
        matrixes = []
        for batch in create_batch_iter(self.valid_data, 1):
            src_words, src_rels, heads, src_lengths, tgt_words = self.pair_data_variable(batch)
            decoder_outputs, attn_scores = self.model.analyze(src_words, src_rels, heads, tgt_words, src_lengths)
            loss, stat = self.model.compute_loss(tgt_words, decoder_outputs)
            matrixes.append(attn_scores)
        return matrixes


    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_rels, heads, src_lengths = self.source_data_variable(batch)
            allHyp, allScores = self.translate_batch(src_words, src_rels, heads, src_lengths)
            all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
            for idx in range(batch_size):
                if all_hyp_inds[idx][-1] == self.tgt_vocab.END_SEQ:
                    all_hyp_inds[idx].pop()
            all_hyp_words = [self.tgt_vocab.id2word(idxs) for idxs in all_hyp_inds]
            for idx, instance in enumerate(batch):
                result['\t'.join(instance[-1])] = all_hyp_words[idx]
        return result


    def translate_batch(self, src_words, src_rels, heads, src_input_lengths):
        beam_size = self.config.beam_size
        encoder_outputs, encoder_hidden = \
            self.model.encode(src_words, src_rels, heads, src_input_lengths)
        decoder_init_hidden = self.model.decoder.init_decoder_state(encoder_hidden)

        context_h = encoder_outputs
        batch_size = context_h.size(1)
        # Expand tensors for each beam.
        context = Variable(context_h.data.repeat(1, beam_size, 1))

        dec_states = Variable(decoder_init_hidden.data.repeat(1, beam_size, 1))

        beam = [Beam(beam_size, self.tgt_vocab, cuda=self.use_cuda) \
                for k in range(batch_size)]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.config.decode_max_time_step):
            input = torch.stack([b.get_current_state() for b in beam if not b.done])
            input = input.t().contiguous().view(1, -1)

            decoder_output, decoder_hidden = self.model.decode(
                Variable(input), context,  dec_states)

            dec_states = [decoder_hidden]

            dec_out = decoder_output.squeeze(0)

            score = self.model.predictor(dec_out)

            out = F.softmax(score, dim=-1).unsqueeze(0)

            word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done: continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]): active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view( \
                         -1, beam_size, remaining_sents, dec_state.size(2))[:, :, idx]
                    sent_states.data.copy_( \
                        sent_states.data.index_select(1, beam[b].get_current_origin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            tt = torch.cuda if self.use_cuda else torch
            active_idx = tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents,  2 * self.config.hidden_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx).view(*new_size))


            dec_states = update_active(dec_states[0])
            dec_out = update_active(dec_out)
            context = update_active(context)

            remaining_sents = len(active)

        #  (4) package everything up
        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        return allHyp, allScores
