import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from module.TreeGRU import *



class EncoderRNN(nn.Module):
    """ The standard RNN encoder. """
    def __init__(self, input_size,
                hidden_size, num_layers=1, 
                dropout=0.1):
        super(EncoderRNN, self).__init__()

        hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.dt_tree = DTTreeGRU(input_size, hidden_size)
        self.td_tree = TDTreeGRU(input_size, hidden_size)
        
    def forward(self, input, heads, lengths=None, hidden=None):
        """ See EncoderBase.forward() for description of args and returns."""
        emb = self.dropout(input)
        #hidden_t = torch.cat([hidden_t[0:hidden_t.size(0):2], hidden_t[1:hidden_t.size(0):2]], 2)
        #outputs = emb.transpose(0, 1)

        max_length, batch_size, input_dim = emb.size()
        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = creatTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs, dt_hidden_ts = self.dt_tree(emb, indexes, trees)
        td_outputs, td_hidden_ts = self.td_tree(emb, indexes, trees)

        outputs = torch.cat([dt_outputs, td_outputs], dim=2).transpose(0, 1)
        output_t = torch.cat([dt_hidden_ts, td_hidden_ts], dim=1).unsqueeze(0)

        #tree_outputs = []
        #tree_hidden_ts = []
        #for idx, head in enumerate(heads):
        #    root, tree = creatTree(head)
        #    root.traverse()
        #    dt_output, dt_hidden_t = self.dt_tree(outputs[idx], root, tree)
        #    td_output, td_hidden_t = self.td_tree(outputs[idx], root, tree)
        #    tree_outputs.append(torch.cat([dt_output, td_output], 1))
        #    tree_hidden_ts.append(torch.cat([dt_hidden_t, td_hidden_t], 0))

        #tree_outputs = torch.stack(tree_outputs, 0)
        #tree_hidden_ts = torch.stack(tree_hidden_ts, 0).unsqueeze(0)

        return outputs, output_t
