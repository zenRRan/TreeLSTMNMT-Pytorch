# -*- coding: utf-8 -*-

import sys
sys.path.extend(["../", "./"])
import os
import re
import time
import pickle
import random
import argparse

import torch
from data.DataLoader import *
from driver.NMTModel import NeuralMT
from driver.NMTHelper import NMTHelper
from driver.Config import Configurable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(nmt, config, model_id, test_files):
    test_start_time = time.time()

    valid_srcs = read_src_corpus(test_files[0])
    eval_data = nmt.prepare_eval_data(valid_srcs)
    result = nmt.translate(eval_data)

    outputFile = test_files[0] + '.' + model_id + '.test'
    output = open(outputFile, 'w', encoding='utf-8')
    ordered_result = []
    for idx, instance in enumerate(eval_data):
        src_key = '\t'.join(instance[-1])
        cur_result = result.get(src_key)
        if cur_result is not None:
            ordered_result.append(cur_result)
        else:
            print("Strange, miss one sentence")
            ordered_result.append([''])

        sentence_out = ' '.join(ordered_result[idx])
        sentence_out = sentence_out.replace(' <unk>', '')

        output.write(sentence_out + '\n')

    output.close()

    command = 'perl %s %s < %s' % (config.bleu_script, ' '.join(test_files[1:]), outputFile)
    bleu_exec = os.popen(command)
    bleu_exec = bleu_exec.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),', bleu_exec, re.S)[0]
    bleu_val = float(bleu_val)

    during_time = float(time.time() - test_start_time)
    print("bleu: %.2f, time %.2f" % (bleu_val, during_time))


if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    # device
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    # parameters
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', type=str, default='default.cfg')
    argparser.add_argument('--model_id', type=str, default='98840')
    argparser.add_argument('--thread', type=int, default=4, help='thread num')
    argparser.add_argument('--use_cuda', action='store_true', default=True)
    argparser.add_argument('--input', type=str, default=None)

    # update parameters
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # load vocab and model
    src_vocab = pickle.load(open(config.save_src_vocab_path, 'rb+'))
    tgt_vocab = pickle.load(open(config.save_tgt_vocab_path, 'rb+'))

    nmt_model = NeuralMT(config, src_vocab.word_size, src_vocab.rel_size, tgt_vocab.size, src_vocab.PAD)
    model_path = config.load_model_path + '.' + args.model_id
    nmt_model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        nmt_model = nmt_model.cuda()

    nmt = NMTHelper(nmt_model, src_vocab, tgt_vocab, config)
    if args.input is None:
        test_files = config.test_files.strip().split(' ')
        test(nmt, config, args.model_id, test_files)
    else:
        with codecs.open(args.input, encoding='utf8') as input_file:
            for line in input_file.readlines():
                test_files = line.strip().split(' ')
                if len(test_files) == 0: continue
                test(nmt, config, args.model_id, test_files)
