# train model on penn treebank language modeling task

import argparse
import numpy as np
import tensorflow as tf
import random
import time
import sys

from math import log, exp
from itertools import product

from rnn import StackRNNCell, StackLSTMCell, RecurrentWrapper
from utils import read_ptb, ptb_chars

# command line arguments

parser = argparse.ArgumentParser(
    description = "Train model on Penn Treebank language modeling task")
parser.add_argument("model", type = str, default = None,
                    help = "rnn | lstm | srnn | slstm")
parser.add_argument("-N", type = int, default = 100, 
                    help = "number of iterations to train model",
                    dest = "n_train_iterations")
parser.add_argument("-S", type = int, default = 1000,
                    help = "number of sequences to train on per iteration",
                    dest = "n_seq_per_iter")
parser.add_argument("-F", type = str, default = None,
                    help = "filename for saved model",
                    dest = "model_ckpt")

args = parser.parse_args()

# train model on ptb

n_train_iterations = args.n_train_iterations
n_seq_per_iter = args.n_seq_per_iter

print("building graph")
with tf.Graph().as_default():

    if args.model == "rnn":
        cell = tf.nn.rnn_cell.BasicRNNCell(100)
    elif args.model == "lstm":
        cell = StackLSTMCell(num_units = 100, no_op = True, n_stack = 10, k = 2)
    elif args.model == "srnn":
        cell = StackRNNCell(num_units = 100, no_op = True, n_stack = 50, k = 2)
    elif args.model == "slstm":
        cell = tf.nn.rnn_cell.BasicLSTMCell(100, state_is_tuple = True)
    else:
        print(parser.print_help())
        sys.exit()

    model_ckpt = args.model_ckpt if args.model_ckpt else args.model + ".ckpt"

    model = RecurrentWrapper(cell, n_symbols = len(ptb_chars), sgd_lr = 0.01)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])

    print("load dataset")
    ptb_train = read_ptb("train")
    ptb_valid = read_ptb("valid")
    ptb_test  = read_ptb("test")

    def lm_perplexity(dataset):
        e = 10 ** -9
        ce_per_char = []
        for ptb_seq in dataset:
            char_probs = sess.run(model.probs,
                feed_dict = { model.symbols : ptb_seq[: -1] })
            ce_per_char += [- log(np.dot(pred_char_dist[0], true_char[0]) + e)
                            for pred_char_dist, true_char in \
                            zip(char_probs, ptb_seq[1 :])]
        return exp(1.0 / len(ce_per_char) * sum(ce_per_char))

    print("training start")
    for i in range(n_train_iterations):
        t = time.time()
        losses = []
        for j in range(n_seq_per_iter):
            ptb_seq = random.choice(ptb_train)
            current_loss, _ = sess.run(
                [model.loss, model.train_op],
                feed_dict = { model.symbols : ptb_seq[: -1],
                              model.targets : ptb_seq[1 :] })
            losses.append(current_loss.mean())

        saver.save(sess, "checkpoints/" + model_ckpt)
        
        print("Number of iterations:", i)
        print(" ", sum(losses) / len(losses))
        print(" time:", time.time() - t)

        t = time.time()
        print(" train pp:", lm_perplexity(ptb_train[: 1000]))
        print(" time:", time.time() - t)
        
        t = time.time()
        print(" valid pp:", lm_perplexity(ptb_valid[: 1000]))
        print(" time:", time.time() - t)

    print(" test pp:", lm_perplexity(ptb_test))
