
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from .conll2002_metrics import *

import numpy as np
from tqdm import tqdm
import os
import logging
logger = logging.getLogger()

index2slot = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime', 'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo', 'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period', 'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference', 'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier', 'B-alarm/recurring_period', 'I-alarm/recurring_period']

class EvaluateTransfer(object):
    def __init__(self, params, dataloader, embedding, n_words):
        self.params = params
        self.embedding = embedding
        self.n_words = n_words
        self.dataloader = dataloader

    def load_best_model(self, path):
        logger.info("Loading best model from %s" % path)
        best_model = torch.load(path)
        self.lstm = best_model["text_encoder"]
        self.intent_predictor = best_model["intent_predictor"]
        self.slot_predictor = best_model["slot_predictor"]
        # change embedding
        self.lstm.embedding = nn.Embedding(self.n_words, self.params.emb_dim)
        self.lstm.embedding.weight.data.copy_(torch.FloatTensor(self.embedding))

        self.lstm, self.intent_predictor, self.slot_predictor = self.lstm.cuda(), self.intent_predictor.cuda(), self.slot_predictor.cuda()

    def evaluate(self):
        # load best model
        path = os.path.join(self.params.dump_path, "best_model.pth")
        self.load_best_model(path)
        
        # evaluate
        self.lstm.eval()
        self.intent_predictor.eval()
        self.slot_predictor.eval()
        intent_pred, slot_pred = [], []
        y1_list, y2_list = [], []
        pbar = tqdm(enumerate(self.dataloader),total=len(self.dataloader))
        for i, (X, lengths, y1, y2) in pbar:
            y1_list.append(y1.data.cpu().numpy())
            y2_list.extend(y2) # y2 is a list
            
            X, lengths = X.cuda(), lengths.cuda()
            lstm_layer = self.lstm(X)
            intent_prediction = self.intent_predictor(lstm_layer, lengths)
            # for intent_pred
            intent_pred.append(intent_prediction.data.cpu().numpy())
            # for slot_pred
            slot_prediction = self.slot_predictor(lstm_layer)
            slot_pred_temp = []
            for i, length in enumerate(lengths):
                slot_pred_each = slot_prediction[i][:length]
                slot_pred_temp.append(slot_pred_each.data.cpu().numpy())
            slot_pred.extend(slot_pred_temp)
        # concatenation
        intent_pred = np.concatenate(intent_pred, axis=0)
        intent_pred = np.argmax(intent_pred, axis=1)
        slot_pred = np.concatenate(slot_pred, axis=0)
        slot_pred = np.argmax(slot_pred, axis=1)
        y1_list = np.concatenate(y1_list, axis=0)
        y2_list = np.concatenate(y2_list, axis=0)
        # evaluate
        intent_acc = accuracy_score(y1_list, intent_pred)

        y2_list = list(y2_list)
        slot_pred = list(slot_pred)
        lines = []
        for pred_index, gold_index in zip(slot_pred, y2_list):
            pred_slot = index2slot[pred_index]
            gold_slot = index2slot[gold_index]
            lines.append("w" + " " + pred_slot + " " + gold_slot)
        results = conll2002_measure(lines)
        slot_f1 = results["fb1"]

        return intent_acc, slot_f1
