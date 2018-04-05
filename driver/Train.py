# -*- coding: utf-8 -*-
import sys
sys.path.extend(["../","./"])
import argparse
import random
from driver.Config import *
from driver.Optim import *
from driver.NMTHelper import *
import pickle
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(nmt, train_srcs, train_tgts, config):
    optimizer = Optim(config)
    optimizer.set_parameters(nmt.model.parameters())
    global_step = 0
    nmt.prepare_training_data(train_srcs, train_tgts)
    valid_files = config.dev_files.strip().split(' ')
    valid_srcs = read_src_corpus(valid_files[0])
    valid_tgts = read_tgt_corpus(valid_files[1])
    nmt.prepare_valid_data(valid_srcs, valid_tgts)

    test_files = config.test_files.strip().split(' ')

    print('start training...')
    best_ppl = 1000000
    best_bleu = -1
    last_bleu = -1
    initial_lr = config.learning_rate
    slow_start = config.slow_start
    for iter in range(config.train_iters):
        # dynamic adjust lr
        total_stats = Statistics()
        batch_num = nmt.batch_num
        batch_iter = 0
        for batch in create_train_batch_iter(nmt.train_data, nmt.batch_size, shuffle=True):
            stat = nmt.train_one_batch(batch)
            total_stats.update(stat)

            batch_iter += 1
            total_stats.print_out(global_step, iter, batch_iter, batch_num)
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                lr = initial_lr * (1 - slow_start)
                slow_start = slow_start * config.slow_start
                optimizer.setRate(lr)
                optimizer.step()
                nmt.model.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                valid_stat = nmt.valid(global_step)
                valid_ppl = valid_stat.ppl()

                dev_start_time = time.time()
                dev_bleu = evaluate(nmt, valid_files, config, global_step)
                during_time = float(time.time() - dev_start_time)
                print("step %d, epoch %d: dev bleu: %.2f, time %.2f" \
                      % (global_step, iter, dev_bleu, during_time))


                test_start_time = time.time()
                test_bleu = evaluate(nmt, test_files, config, global_step)
                during_time = float(time.time() - test_start_time)
                print("step %d, epoch %d: dev bleu: %.2f, time %.2f" \
                      % (global_step, iter, test_bleu, during_time))

                not_saved = True
                if dev_bleu > best_bleu:
                    print("Exceed best bleu: history = %.2f, current = %.2f" % (best_bleu, dev_bleu))
                    best_bleu = dev_bleu
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(nmt.model.state_dict(), config.save_model_path + '.' + str(global_step))
                        not_saved = False

                if dev_bleu < last_bleu:
                    initial_lr = initial_lr * config.decay
                    print("Decaying initial learning rate to %.6f" % initial_lr)
                last_bleu = dev_bleu

                if valid_ppl < best_ppl:
                    print("Exceed best ppl: history = %.2f, current = %.2f" % (best_ppl, valid_ppl))
                    best_ppl = valid_ppl
                    if not_saved and config.save_after > 0 and iter > config.save_after:
                        torch.save(nmt.model.state_dict(), config.save_model_path + '.' + str(global_step))
                        not_saved = False


def evaluate(nmt, eval_files, config, global_step):
    valid_srcs = read_src_corpus(eval_files[0])
    eval_data = nmt.prepare_eval_data(valid_srcs)
    result = nmt.translate(eval_data)

    outputFile = eval_files[0] + '.' + str(global_step)
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

    command = 'perl %s %s < %s' % (config.bleu_script, ' '.join(eval_files[1:]), outputFile)
    bleu_exec = os.popen(command)
    bleu_exec = bleu_exec.read()
    # Get bleu value
    bleu_val = re.findall('BLEU = (.*?),', bleu_exec, re.S)[0]
    bleu_val = float(bleu_val)

    return bleu_val



if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--tgt_word_file', default=None)
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    train_files = config.train_files.strip().split(' ')
    train_srcs, train_tgts = read_training_corpus(train_files[0], train_files[1], config.max_train_length)

    src_vocab, tgt_vocab = creat_vocabularies(train_srcs, train_tgts, config.vocab_size, config.vocab_size)
    if args.tgt_word_file is not None:
        tgt_words = read_tgt_words(args.tgt_word_file)
        tgt_vocab = TGTVocab(tgt_words)
    pickle.dump(src_vocab, open(config.save_src_vocab_path, 'wb'))
    pickle.dump(tgt_vocab, open(config.save_tgt_vocab_path, 'wb'))

    print("Sentence Number: #train = %d" %(len(train_srcs)))

    # model
    nmt_model = NeuralMT(config, src_vocab.word_size, src_vocab.rel_size, tgt_vocab.size, src_vocab.PAD)
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        nmt_model = nmt_model.cuda()

    nmt = NMTHelper(nmt_model, src_vocab, tgt_vocab, config)

    train(nmt, train_srcs, train_tgts, config)

