# train model on algorithmic patterns

import argparse
import numpy as np
import tensorflow as tf

from itertools import product

from rnn import StackRNNCell, StackLSTMCell, RecurrentWrapper
from sequences import SeqGenerator

# command line arguments

parser = argparse.ArgumentParser(
    description = "Train model on algorithmic patterns")
parser.add_argument("model", type = str, default = None,
                    help = "rnn | lstm | srnn | slstm")
parser.add_argument("task", type = int, default = None,
                    help = "1 | 2 | 3 | 4 | 5")
parser.add_argument("n_symbols", type = int, default = None,
                    help = "number of symbols in alphabet")
parser.add_argument("-N", type = int, default = 10000 // 100, 
                    help = "number of iterations to train model",
                    dest = "n_iter")
parser.add_argument("-S", type = int, default = 100,
                    help = "number of sequences to train on per iteration",
                    dest = "n_seq_per_iter")

args = parser.parse_args()

# hyperparameter search on algorithmic patterns

n_iterations = args.n_iter
n_seq_per_iter = args.n_seq_per_iter

assert args.task in [1, 2, 3, 4, 5]
task = args.task
n_symbols = args.n_symbols
n_min = 2
n_max_train = 15
n_max_val = 19
n_max_test = 60

if args.model == "rnn" or args.model == "lstm":
    hidden_sizes = [50, 100, 200]
    hidden_layers = [1, 2]
    sgd_lrs = [0.1, 0.01, 0.001]
    n_stacks = [0]
    ks = [0]
elif args.model == "srnn" or args.model == "slstm":
    hidden_sizes = [20, 40, 100]
    hidden_layers = [1]
    sgd_lrs = [0.1, 0.01, 0.001]
    n_stacks = [1, 2, 5, 10]
    ks = [1, 2]
else:
    print(parser.print_help())
    sys.exit()

for hidden_size, n_layers, sgd_lr, n_stack, k in \
    product(hidden_sizes, hidden_layers, sgd_lrs, n_stacks, ks):
    print("=" * 60)
    print(" units: ", hidden_size)
    print(" layers: ", n_layers)
    print(" lr: ", sgd_lr)
    print(" stacks: ", n_stack)
    print(" k: ", k)
    print("=" * 60)
    
    # set up session
    with tf.Graph().as_default():
        if args.model == "rnn":
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicRNNCell(hidden_size)
                 for _ in range(n_layers)])
        elif args.model == "lstm":
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                 for _ in range(n_layers)])
        elif args.model == "srnn":
            cell = StackRNNCell(hidden_size, no_op = False,
                                n_stack = n_stack, k = k)
        elif args.model == "slstm":
            cell = StackLSTMCell(hidden_size, no_op = False,
                                 n_stack = n_stack, k = k)
        model = RecurrentWrapper(cell, n_symbols = n_symbols, sgd_lr = sgd_lr)

        sess = tf.Session()
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        n_max_train_ = 5
        for i in range(n_iterations):
            if i % 5 == 4:
                n_max_train_ = min(n_max_train_ + 1, n_max_train)
            gen = SeqGenerator(1, n_symbols, n_min = n_min,
                               n_max = n_max_train_)
            gen_ = SeqGenerator(1, n_symbols, n_min = n_min,
                                n_max = n_max_val, sequential = True)
            gen_test = SeqGenerator(1, n_symbols, n_min = n_min,
                                    n_max = n_max_test, sequential = True)
            losses = []
            for j in range(n_seq_per_iter):
                symbols_batch, targets_batch, _ = gen.next_sequence()
                current_loss, _ = sess.run(
                    [model.loss, model.train_op],
                    feed_dict = { model.symbols : symbols_batch,
                                  model.targets : targets_batch })
                losses.append(current_loss.mean())
                
            print("Number of iterations:", i)
            print(" ", sum(losses) / len(losses))

            # calculate accuracy for train + validation set
            sequences_correct = []
            for n in range(n_min, n_max_val + 1):
                # is_pred signifies if next symbol is predictable
                symbols_batch_, targets_batch_, is_pred_ = \
                                gen_.next_sequence()
                
                predictions = sess.run(model.preds,
                    feed_dict = { model.symbols : symbols_batch_ })

                correct = all([np.argmax(t[0]) == p[0] for t, p, i in \
                               zip(targets_batch_, predictions, is_pred_)
                               if i is True])
                sequences_correct.append(correct)
            print(" ", sum(sequences_correct) / len(sequences_correct))

            # calculate accuracy for train + validation + test set
            sequences_correct = []
            for n in range(n_min, n_max_test + 1):
                symbols_batch_, targets_batch_, is_pred_ = \
                                gen_test.next_sequence()
                
                predictions = sess.run(model.preds,
                    feed_dict = { model.symbols : symbols_batch_ })

                correct = all([np.argmax(t[0]) == p[0] for t, p, i in \
                               zip(targets_batch_, predictions, is_pred_)
                               if i is True])
                sequences_correct.append(correct)
            print(" ", sum(sequences_correct) / len(sequences_correct))
