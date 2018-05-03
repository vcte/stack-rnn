# utility functions

import numpy as np

def map_one_hot(seq, n_symbols):
    a = np.zeros((len(seq), 1, n_symbols), dtype = np.float32)
    a[np.arange(len(seq)), :, seq] = 1
    return a

def read_csv(fname, sep = "\t", typ = int):
    data = []
    with open(fname, mode = "r", encoding = "utf-8") as f:
        for line in f.readlines():
            data.append(list(map(typ, line.split(sep))))
    return data

ptb_chars = ['\n', '#', '$', '&', "'", '*', '-', '.', '/',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             '<', '>', 'N', '\\', '_', 'a', 'b', 'c', 'd', 'e',
             'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
             'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_code = { c : ptb_chars.index(c) for c in ptb_chars}

ptb_dir = "simple-examples/simple-examples/data/"
def read_ptb(dataset = "train"):
    # dataset = "train" | "valid" | "test"
    data = read_csv(ptb_dir + "ptb.char.%s.txt" % dataset, sep = " ", typ = str)
    data = [[char_to_code[char] for char in line if char != '']
            for line in data]
    data = [line for line in data if len(line) > 0]
    data = [map_one_hot(line, n_symbols = len(ptb_chars)) for line in data]
    return data
