
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from .lvm import LVM
from .utils import load_embedding
from .crf import *

SLOT_PAD = 0

class Lstm(nn.Module):
    def __init__(self, params, vocab):
        super(Lstm, self).__init__()
        self.n_layer = params.n_layer
        self.n_words = vocab.n_words
        self.emb_dim = params.emb_dim
        self.hidden_dim = params.hidden_dim
        self.dropout = params.dropout
        self.bidirection = params.bidirection
        self.emb_file = params.emb_file_en
        self.freeze_emb = params.freeze_emb
        self.transfer = params.transfer
        self.embnoise = params.embnoise

        # embedding layer
        self.embedding = nn.Embedding(self.n_words, self.emb_dim)
        # load embedding
        embedding = load_embedding(vocab, self.emb_dim, self.emb_file)
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))
        
        # LSTM layers
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_layer, 
                        dropout=self.dropout, bidirectional=self.bidirection, batch_first=True)
    
    def forward(self, x):
        """
        Input:
            x: text (bsz, seq_len)
        Output:
            last_layer: last layer of lstm (bsz, seq_len, hidden_dim)
        """
        embeddings = self.embedding(x)
        embeddings = embeddings.detach() if self.freeze_emb else embeddings
        if self.embnoise == True and self.transfer == False and self.training == True:
            size = embeddings.size()
            noise = torch.randn(size) * 0.05
            noise = noise.cuda()
            embeddings += noise
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        # LSTM
        # lstm_output (batch_first): (bsz, seq_len, hidden_dim)
        lstm_output, (_, _) = self.lstm(embeddings)

        return lstm_output

# https://github.com/huggingface/torchMoji/blob/master/torchmoji/attlayer.py
class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    """
    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        torch.nn.init.uniform(self.attention_vector.data, -0.01, 0.01)

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()
        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        idxes = torch.arange(
            0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        idxes = idxes.cuda()
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return (representations, attentions if self.return_attention else None)

class IntentPredictor(nn.Module):
    def __init__(self, params):
        super(IntentPredictor, self).__init__()
        self.num_intent = params.num_intent
        self.attention_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.atten_layer = Attention(attention_size=self.attention_size, return_attention=False)

        self.lvm = params.lvm
        self.lvm_dim = params.lvm_dim
        if self.lvm == True:
            self.lvm_layer = LVM(params)
            self.linear = nn.Linear(self.lvm_dim, self.num_intent)
        else:
            self.linear = nn.Linear(self.attention_size, self.num_intent)
        
        self.transfer = params.transfer
    
    def forward(self, inputs, lengths):
        """ forward pass
        Inputs:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
            lengths: lengths of x (bsz, )
        Output:
            prediction: Intent prediction (bsz, num_intent)
        """
        atten_layer, _ = self.atten_layer(inputs, lengths)
        if self.lvm == True:
            # lvm_layer, norm_kl = self.lvm_layer(atten_layer)
            lvm_layer = self.lvm_layer(atten_layer)
            prediction = self.linear(lvm_layer)
        else:
            prediction = self.linear(atten_layer)

        return prediction
    
    def out_lvm_layer(self, inputs, lengths):
        assert self.lvm == True
        atten_layer, _ = self.atten_layer(inputs, lengths)
        lvm_layer = self.lvm_layer(atten_layer)
        return lvm_layer


class SlotPredictor(nn.Module):
    def __init__(self, params):
        super(SlotPredictor, self).__init__()
        self.num_slot = params.num_slot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim

        self.lvm = params.lvm
        self.lvm_dim = params.lvm_dim
        if self.lvm == True:
            self.lvm_layer = LVM(params)
            self.linear = nn.Linear(self.lvm_dim, self.num_slot)
        else:
            self.linear = nn.Linear(self.hidden_dim, self.num_slot)

        self.crf = params.crf
        if self.crf == True:
            self.crf_layer = CRF(self.num_slot)

        self.transfer = params.transfer
    
    def forward(self, inputs):
        """ forward pass
        Input:
            inputs: lstm hidden layer (bsz, seq_len, hidden_dim)
        Output:
            prediction: slot prediction (bsz, seq_len, num_slot)
        """
        if self.lvm == True:
            lvm_layer = self.lvm_layer(inputs)
            prediction = self.linear(lvm_layer)
        else:
            prediction = self.linear(inputs)

        return prediction

    def out_lvm_layer(self, inputs):
        assert self.lvm == True
        lvm_layer = self.lvm_layer(inputs)
        return lvm_layer

    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        mask = self.make_mask(lengths)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: output of SlotPredictor (bsz, seq_len, num_slot)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        mask = self.make_mask(lengths)
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction

    def make_mask(self, lengths):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        mask = torch.LongTensor(bsz, max_len).fill_(1)
        for i in range(bsz):
            length = lengths[i]
            mask[i, length:max_len] = 0
        mask = mask.cuda()
        return mask
    
    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y

