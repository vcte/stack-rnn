# sequence generator class

import random

from utils import map_one_hot

class SeqGenerator:
    def __init__(self, task, n_symbols, n_min, n_max,
                 sequential = False):
        """Initialize sequence generator.

        Args:
          task: int, task = 1 for counting tasks
                     task = 2 for count + double task
                     task = 3 for count + add task
                     task = 4 for memorization task
                     task = 5 for binary addition task
          n_symbols: int, number of symbols, only relevant when task = 1
          n_min: int, min length of sequences
          n_max: int, max length of sequences
          sequential: bool, whether sequences are generated in order or randomly
        """
        self.task = task
        self.n_symbols = n_symbols
        self.n_min = n_min
        self.n_max = n_max
        self.sequential = sequential

        self.seq = []
        self.is_pred = []

        if sequential:
            self.current_n = n_min

    def gen_sequence_1(self, n = None):
        # generate patterns of form a^n b^n c^n ...
        if n is None:
            n = random.randint(self.n_min, self.n_max)
        
        seq = []
        is_pred = []
        for i, symb in enumerate(range(self.n_symbols)):
            seq += [symb for _ in range(n)]

            # only parts of sequence after a^n are predictable
            if i == 0:
                is_pred += [False] * n
            else:
                is_pred += [True] * n
        return seq, is_pred

    def gen_sequence_2(self, n = None):
        # generates patterns of the form a^n b^2n
        assert self.n_symbols == 2
        if n is None:
            n = random.randint(self.n_min, self.n_max)

        seq = []
        is_pred = []
        for i, symb in enumerate(range(self.n_symbols)):
            seq += [symb for _ in range(n * (i + 1))]

            # only parts of sequence after a^n are predictable
            if i == 0:
                is_pred += [False] * (n * (i + 1))
            else:
                is_pred += [True] * (n * (i + 1))
        return seq, is_pred

    def gen_sequence_3(self, n = None):
        # generates patterns of the form a^(n - m) b^m c^n
        assert self.n_symbols == 3
        if n is None:
            n = random.randint(self.n_min, self.n_max)
        m = random.randint(1, n - 1)

        seq = []
        is_pred = []

        seq += [0 for _ in range(n - m)]
        is_pred += [False for _ in range(n - m)]

        seq += [1 for _ in range(m)]
        is_pred += [False for _ in range(m)]

        seq += [2 for _ in range(n)]
        is_pred += [True for _ in range(n)]

        return seq, is_pred

    def gen_sequence_4(self, n = None):
        # generates patterns of the form {bc}^n a {bc}^n
        # where {bc}^n is same random string and 'a' is a separator token
        assert self.n_symbols == 3
        if n is None:
            n = random.randint(self.n_min, self.n_max)

        random_bitstring = [random.choice((1, 2)) for _ in range(n)]

        seq = random_bitstring + [0] + random_bitstring
        is_pred = [False for _ in range(n)] + [True for _ in range(n + 1)]

        return seq, is_pred

    def gen_sequence_5(self, n = None):
        # binary addition task
        # generates patterns of form x + y = [x + y]
        # where x = {ab}^(n - m), y = {ab}^m
        assert self.n_symbols == 4
        if n is None:
            n = random.randint(self.n_min, self.n_max)
        m = random.randint(1, n - 1)

        bs_to_int = lambda s: int("".join(map(str, s)), base = 2)
        int_to_bs = lambda i: list(map(int, bin(i)[2:]))
        
        x = [1] + [random.choice((0, 1)) for _ in range(n - m - 1)]
        y = [1] + [random.choice((0, 1)) for _ in range(m - 1)]
        xpy = list(reversed(int_to_bs(bs_to_int(x) + bs_to_int(y))))

        seq = x + [2] + y + [3] + xpy
        is_pred = [False for _ in range(n + 1)] + \
                  [True for _ in range(1 + len(xpy))]

        return seq, is_pred

    def gen_sequence(self):
        if self.sequential:
            n = self.current_n
            self.current_n += 1
            if self.current_n > self.n_max:
                self.current_n = self.n_min
        else:
            n = None
        
        if self.task == 1:
            return self.gen_sequence_1(n)
        if self.task == 2:
            return self.gen_sequence_2(n)
        if self.task == 3:
            return self.gen_sequence_3(n)
        if self.task == 4:
            return self.gen_sequence_4(n)
        if self.task == 5:
            return self.gen_sequence_5(n)
        else:
            raise ValueError("Unrecognized task: %d" % task)

    def next_n_sequences_chunked(self, n, num_steps):
        for _ in range(n):
            seq_, is_pred_ = self.gen_sequence()
            self.seq += seq_
            self.is_pred += is_pred_

            while len(self.seq) >= num_steps + 1:
                # values to return
                chunk = self.seq[0 : num_steps]
                target = self.seq[1 : num_steps + 1]
                is_pred = self.is_pred[0 : num_steps]

                # update running sequence
                self.seq = self.seq[num_steps :]
                self.is_pred = self.is_pred[num_steps :]

                yield map_one_hot(chunk, self.n_symbols), \
                      map_one_hot(target, self.n_symbols), \
                      is_pred

        if len(self.seq) > 0:
            # start of next sequence is always 'a'
            yield map_one_hot(self.seq, self.n_symbols), \
                  map_one_hot(self.seq[1 :] + [0], self.n_symbols), \
                  is_pred

    def next_sequence(self):
        seq, is_pred = self.gen_sequence()

        return map_one_hot(seq, self.n_symbols), \
               map_one_hot(seq[1 :] + [0], self.n_symbols), \
               is_pred
