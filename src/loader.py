
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from .preprocess import preprocess, PAD_INDEX

import logging
logger = logging.getLogger()

class Dataset(data.Dataset):
    def __init__(self, data):
        self.X = data["text"]
        self.y1 = data["intent"]
        self.y2 = data["slot"]

    def __getitem__(self, index):
        return self.X[index], self.y1[index], self.y2[index] 
    
    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y1, y2 = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    y1 = torch.LongTensor(y1)
    return padded_seqs, lengths, y1, y2

def load_data(clean_txt = True):
    data = {"en": {}, "es": {}, "th": {}}
    # load English data
    preprocess(data, "en", clean_txt)
    # load Spanish data
    preprocess(data, "es", clean_txt)
    # load Thai data
    preprocess(data, "th", clean_txt)

    return data

def get_dataloader(params, lang):
    data = load_data(clean_txt=params.clean_txt)
    dataset_tr = Dataset(data[lang]["train"])
    dataset_val = Dataset(data[lang]["eval"])
    dataset_test = Dataset(data[lang]["test"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return dataloader_tr, dataloader_val, dataloader_test, data[lang]["vocab"]
