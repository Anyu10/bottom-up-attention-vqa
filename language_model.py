import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        #self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken) #? why there is a padding_index
        #* Embedding is simply a lookup table, its shape is (ntoken+1, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        p = torch.zeros((1, weight_init.shape[1]))
        weight_init = torch.cat([weight_init, p], dim=0)
        assert weight_init.shape == (self.ntoken+1, self.emb_dim)
        self.emb = nn.Embedding.from_pretrained(weight_init, padding_idx=self.ntoken)
        # self.emb.weight.data[:ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch, deactivate):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())
    #? ^ make no sense

    def init_hidden(self, batch):
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == "LSTM":
            return (torch.zeros(hid_shape, dtype=torch.float32, device='cuda', requires_grad=True),
                    torch.zeros(hid_shape, dtype=torch.float32, device='cuda', requires_grad=True))
        else:
            return torch.zeros(hid_shape, dtype=torch.float32, device='cuda', requires_grad=True)

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()   #* Resets parameter data pointer so that they can use faster code paths.
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output
