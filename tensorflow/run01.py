# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""
import sys
if sys.version[0] == '2':
    # reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
from rc_model01 import RCModel
import tensorflow as tf

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=16,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=999999,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=200,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=100,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=300,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/DuReader_preprocessed/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/DuReader_preprocessed/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/DuReader_preprocessed/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/DuReader_preprocessed/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/DuReader_preprocessed_test/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/DuReader_preprocessed/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/DuReader_preprocessed/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    ####################可加入预训练词向量##################
    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')

config = {
    'model_name':'BiDAF'
}
def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')

    ####重新加载模型进行训练
    file_pre = args.model_dir
    with tf.Session() as sess:
        saver = tf.train.Saver()
        try:
            saver.restore(sess, file_pre + config['model_name'])
        except:
            pass

        def train_epoch(train_batches, dropout_keep_prob):
            total_num, total_loss = 0, 0
            log_every_n_batch, n_batch_loss = 50, 0
            for bitx, batch in enumerate(train_batches, 1):
                feed_dict = {rc_model.p: batch['passage_token_ids'],
                             rc_model.q: batch['question_token_ids'],
                             rc_model.p_length: batch['passage_length'],
                             rc_model.q_length: batch['question_length'],
                             rc_model.start_label: batch['start_id'],
                             rc_model.end_label: batch['end_id'],
                             rc_model.dropout_keep_prob: dropout_keep_prob}
                _, loss = rc_model.sess.run([rc_model.train_op, rc_model.loss], feed_dict)
                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])
                n_batch_loss += loss
                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    rc_model.logger.info('Average loss from batch {} to {} is {}'.format(
                        bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                    n_batch_loss = 0
            return 1.0 * total_loss / total_num

        def train(self, data, epochs, batch_size, file_pre, config,
                  dropout_keep_prob=1.0, evaluate=True):
            pad_id = self.vocab.get_id(self.vocab.pad_token)
            max_bleu_4 = 0
            for epoch in range(1, epochs):
                rc_model.logger.info('Training the model for epoch {}'.format(epoch))
                train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
                train_loss = train_epoch(train_batches, dropout_keep_prob)
                rc_model.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

                if evaluate:
                    logger.info('Evaluating the model after epoch {}'.format(epoch))
                    if data.dev_set is not None:
                        eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                        eval_loss, bleu_rouge = rc_model.evaluate(eval_batches)
                        logger.info('Dev eval loss {}'.format(eval_loss))
                        rc_model.logger.info('Dev eval result: {}'.format(bleu_rouge))

                        if bleu_rouge['Bleu-4'] > max_bleu_4:
                            rc_model.save(file_pre + config['model_name'])
                            max_bleu_4 = bleu_rouge['Bleu-4']
                    else:
                        rc_model.logger.warning('No dev set is loaded for evaluation in the dataset!')
                else:
                    # self.save(save_dir, save_prefix + '_' + str(epoch))不保存每一个轮次的校验结果
                    rc_model.save(file_pre + config['model_name'])


        train(brc_data, args.epochs, args.batch_size, file_pre,
                       config,
                       dropout_keep_prob=args.dropout_keep_prob)
        logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    # rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)

    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    # rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    ################################
    file_pre = args.model_dir
    saver = tf.train.Saver()
    saver.restore(file_pre + config['model_name'])
    ######################################################
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
