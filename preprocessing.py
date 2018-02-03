import os
import argparse
import codecs
import torch
import mmap
import numpy as np
import itertools

from collections import Counter
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--corpus', type=str, default='./data/text8', help="corpus path")
    parser.add_argument('--unk', type=str, default='<unk>', help="UNK token")
    parser.add_argument('--window', type=int, default=7, help="window size")
    parser.add_argument('--max_vocab', type=int, default=200000, help="maximum number of vocab")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size of the dataset")
    parser.add_argument('--n_negs', type=int, default=20, help="negative samples")
    return parser.parse_args()


class dataset(object):
    def __init__(self, args):
        self.window = args.window
        self.unk = args.unk
        self.data_dir = args.data_dir
        self.corpus = args.corpus
        self.max_vocab = args.max_vocab
        self.batch_size = args.batch_size
        self.n_negs = args.n_negs

    def skipgram(self, sentence, i):
        inword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return inword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(
            self.window - len(right))]

    def build_dataset(self):
        print("read dataset...")
        step = 0
        self.wc = {self.unk: 1}
        sent = []
        with open(self.corpus, 'rb') as file:
            for line in file:
                try:
                    sent.append(line.decode('utf-8'))
                except:
                    pass
        print(len(sent))
        sent = [line.strip().lower().split() for line in sent]
        sent = list(itertools.chain(*sent))

        print("build vocabulary...")
        word_count = Counter(sent)
        self.stoi = {k: idx for idx, (k, _) in enumerate(word_count.most_common(self.max_vocab) + [(self.unk, 1)])}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab = set([word for word in self.stoi])
        freq_count = np.array(
            [v for (k, v) in word_count.most_common(self.max_vocab) + [(self.unk, 1)] if k in self.vocab])
        freq_power = np.power(freq_count, 0.75)
        freq_power /= freq_power.sum()
        self.weights = torch.FloatTensor(freq_power)
        print("build dataset...")

        self.data = []
        sent = [self.stoi[x] if x in self.vocab else self.stoi[self.unk] for x in sent]
        stride = self.window // 2 + self.window % 2
        for i in tqdm(range(len(sent) // stride - self.window)):
            iword, owords = self.skipgram(sent, i * stride + self.window)
            if (iword != self.unk) & all([oword != self.unk for oword in owords]):
                self.data.append([[iword, oword] for oword in owords])
        self.data = list(itertools.chain(*self.data))
        self.data = np.array(self.data).reshape(-1, 2)
        i_data, o_data = torch.from_numpy(np.array(self.data)[:, 0]), torch.from_numpy(np.array(self.data)[:, 1])
        self.dataset = TensorDataset(i_data, o_data)
        self.dsetIter = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)