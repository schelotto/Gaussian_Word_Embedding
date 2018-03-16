import torch
import math
import argparse
import os

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import itertools

from torch.autograd import Variable
from preprocessing import dataset
from preprocessing import parse_args
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--embed_dim', type=int, default=100, help="dimension of the word embedding")
parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
parser.add_argument('--C', type=float, default=1.0, help="C")
parser.add_argument('--sigma_min', type=float, default=0.1, help="minimum of the sigma")
parser.add_argument('--sigma_max', type=float, default=10.0, help="maximum of the sigma")
parser.add_argument('--ob', type=float, default=1.0, help="objective bound")
args = parser.parse_args()

class GaussianEmbedding(nn.Module):
    def __init__(self, args):
        super(GaussianEmbedding, self).__init__()
        self.embed_dim = args.embed_dim
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.C = args.C
        self.ob = args.ob
        self.dset = args.dset
        self.dset.build_dataset()
        self.vocab_size = len(self.dset.vocab)

        # Model
        self.mu = nn.Embedding(self.vocab_size + 1, self.embed_dim)
        self.log_sigma = nn.Embedding(self.vocab_size + 1, self.embed_dim)

        self.mu_pos = nn.Embedding(self.vocab_size +1, self.embed_dim)
        self.log_sigma_pos = nn.Embedding(self.vocab_size + 1, self.embed_dim)

        self.mu_neg = nn.Embedding(self.vocab_size + 1, self.embed_dim)
        self.log_sigma_neg = nn.Embedding(self.vocab_size + 1, self.embed_dim)

    def kl_energy(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """

        assert mu_i.size()[0] == mu_j.size()[0]

        det_fac = torch.sum(torch.log(sigma_i), 1) - torch.sum(torch.log(sigma_j), 1)
        trace_fac = torch.sum(sigma_j / sigma_i, 1)
        diff_mu = torch.sum((mu_i - mu_j) ** 2 / sigma_i, 1)
        return -0.5 * (trace_fac - det_fac + diff_mu + self.embed_dim)

    def el_energy(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """

        assert mu_i.size()[0] == mu_j.size()[0]

        det_fac = torch.sum(torch.log(sigma_i + sigma_j), 1)
        diff_mu = torch.sum((mu_i - mu_j) ** 2 / (sigma_j + sigma_i), 1)
        return torch.exp(-0.5 * (det_fac + diff_mu + self.embed_dim * torch.log(2 * math.pi)))

    def wd_energy(self, mu_i, mu_j, sigma_i, sigma_j):
        """
        :param mu_i: mu of word i: [batch, embed]
        :param mu_j: mu of word j: [batch, embed]
        :param sigma_i: sigma of word i: [batch, embed]
        :param sigma_j: sigma of word j: [batch, embed]
        :return: the energy function between the two batchs of  data: [batch]
        """
        assert mu_i.size()[0] == mu_j.size()[0]

        mu_diff = torch.sum((mu_i - mu_j) ** 2, 1)
        sigma_diff = torch.sum((sigma_i + sigma_j - 2 * (sigma_i * sigma_j) ** 0.5), 1)
        return -(mu_diff + sigma_diff) ** 0.5

    def forward(self, words_i, words_j):
        batch_size = words_i.size()[0]

        for p in itertools.chain(self.log_sigma.parameters(),
                                 self.log_sigma_neg.parameters(),
                                 self.log_sigma_pos.parameters()):
            p.data.clamp_(math.log(self.sigma_min), math.log(self.sigma_max))

        for p in itertools.chain(self.mu.parameters(),
                                 self.mu_neg.parameters(),
                                 self.mu_pos.parameters()):
            p.data.clamp_(-math.sqrt(self.C), math.sqrt(self.C))

        words_n = torch.multinomial(self.dset.weights, batch_size, replacement=True)
        words_n = Variable(words_n).cuda()

        mu_i, mu_j, mu_n = self.mu(words_i), self.mu_pos(words_j), self.mu_neg(words_n)
        sigma_i, sigma_j, sigma_n = torch.exp(self.log_sigma(words_i)), \
                                    torch.exp(self.log_sigma_pos(words_j)), \
                                    torch.exp(self.log_sigma_neg(words_n))

        return torch.mean(F.relu(self.ob - self.kl_energy(mu_i, mu_j, sigma_i, sigma_j) + self.kl_energy(mu_i, mu_n, sigma_i, sigma_n)), dim=0)

    def nn(self, word, k):
        embedding = self.mu.weight.data.cpu() # [dict, embed_size]
        vector = embedding[self.dset.stoi[word], :].view(-1, 1) # [embed_size, 1]
        distance = torch.mm(embedding, vector).squeeze() / torch.norm(embedding, 2, 1)
        distance = distance / torch.norm(vector, 2, 0)[0]
        distance = distance.numpy()
        index = np.argsort(distance)[:-k]
        return [self.dset.itos[x] for x in index]

args.dset = dataset(parse_args())
g_emb = GaussianEmbedding(args)
g_emb = g_emb.cuda()
optimizer = torch.optim.Adam(g_emb.parameters(), lr=0.001)

global_step = 0
for epoch in range(args.epochs):
    step = 0
    for (words_i, words_j) in tqdm(g_emb.dset.dsetIter):
        optimizer.zero_grad()
        words_i = Variable(words_i).cuda()
        words_j = Variable(words_j).cuda()
        loss = g_emb(words_i, words_j)
        loss.backward()
        optimizer.step()

        step += 1
        global_step += 1

        if (global_step + 1) % (len(g_emb.dset.dsetIter) // 10) == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.2f' % (
            epoch + 1, args.epochs, step + 1, len(g_emb.dset.dsetIter), loss.data[0]))

    word_embedding = g_emb.mu.weight.cpu().data.numpy()
    with open(os.path.join('embedding', 'word_embedding.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join([str(len(g_emb.dset.itos)-1), str(args.embed_dim)]) + '\n')
        for i in range(len(g_emb.dset.itos)):
            if g_emb.dset.itow[i] != '<pad>':
                embed_i = [g_emb.dset.itos[i]] + list(map(lambda x: '%.5f' % (x), word_embedding[i, :]))
                f.write(' '.join(embed_i))
                f.write('\n')