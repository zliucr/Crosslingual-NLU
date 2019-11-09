
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
import logging
import os

from sklearn.metrics import accuracy_score
from .conll2002_metrics import *

logger = logging.getLogger()

index2slot = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime', 'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo', 'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period', 'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference', 'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier', 'B-alarm/recurring_period', 'I-alarm/recurring_period']

class DialogTrainer(object):
    def __init__(self, params, lstm, intent_predictor, slot_predictor):
        self.lstm = lstm
        self.intent_predictor = intent_predictor
        self.slot_predictor = slot_predictor
        self.lr = params.lr
        self.params = params
        
        model = [
            {"params": self.lstm.parameters()},
            {"params": self.intent_predictor.parameters()},
            {"params": self.slot_predictor.parameters()}
        ]
        if params.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model, lr=self.lr)
        elif params.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(model, lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_intent_acc = 0
        self.best_slot_f1 = 0
        
        self.stop_training_flag = False

    def train_step(self, epoch, X, lengths, y1, y2):
        self.lstm.train()
        self.intent_predictor.train()
        self.slot_predictor.train()

        lstm_layer = self.lstm(X)
        intent_prediction = self.intent_predictor(lstm_layer, lengths)
        
        # train IntentPredictor
        self.optimizer.zero_grad()
        intent_loss = self.loss_fn(intent_prediction, y1)
        intent_loss.backward(retain_graph=True)
        self.optimizer.step()

        # train SlotPredictor
        slot_prediction = self.slot_predictor(lstm_layer)
        
        if self.params.crf == True:
            slot_loss = self.slot_predictor.crf_loss(slot_prediction, lengths, y2)
            self.optimizer.zero_grad()
            slot_loss.backward()
            self.optimizer.step()

            return intent_loss.item(), slot_loss.item()

        else:
            slot_loss_list = []
            self.optimizer.zero_grad()
            for i, length in enumerate(lengths):
                slot_pred_each = slot_prediction[i][:length]
                slot_loss = self.loss_fn(slot_pred_each, torch.LongTensor(y2[i]).cuda())
                slot_loss.backward(retain_graph=True)
                slot_loss_list.append(slot_loss.item())
            self.optimizer.step()
        
            return intent_loss.item(), np.mean(slot_loss_list)

    def evaluate(self, dataloader, istestset=False):
        if istestset == True:
            # load best model
            best_model_path = os.path.join(self.params.dump_path, "best_model.pth")
            logger.info("Loading best model from %s" % best_model_path)
            best_model = torch.load(best_model_path)
            self.lstm = best_model["text_encoder"]
            self.intent_predictor = best_model["intent_predictor"]
            self.slot_predictor = best_model["slot_predictor"]
        self.lstm.eval()
        self.intent_predictor.eval()
        self.slot_predictor.eval()
        intent_pred, slot_pred = [], []
        y1_list, y2_list = [], []
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))
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
            if self.params.crf == True:
                slot_pred_batch = self.slot_predictor.crf_decode(slot_prediction, lengths)
                slot_pred.extend(slot_pred_batch)
            else:
                slot_pred_temp = []
                for i, length in enumerate(lengths):
                    slot_pred_each = slot_prediction[i][:length]
                    slot_pred_temp.append(slot_pred_each.data.cpu().numpy())
                slot_pred.extend(slot_pred_temp)
        # concatenation
        intent_pred = np.concatenate(intent_pred, axis=0)
        intent_pred = np.argmax(intent_pred, axis=1)
        slot_pred = np.concatenate(slot_pred, axis=0)
        if self.params.crf == False:
            slot_pred = np.argmax(slot_pred, axis=1)
        y1_list = np.concatenate(y1_list, axis=0)
        y2_list = np.concatenate(y2_list, axis=0)
        intent_acc = accuracy_score(y1_list, intent_pred)

        # calcuate f1 score
        y2_list = list(y2_list)
        slot_pred = list(slot_pred)
        lines = []
        for pred_index, gold_index in zip(slot_pred, y2_list):
            pred_slot = index2slot[pred_index]
            gold_slot = index2slot[gold_index]
            lines.append("w" + " " + pred_slot + " " + gold_slot)
        results = conll2002_measure(lines)
        slot_f1 = results["fb1"]

        if intent_acc > self.best_intent_acc:
            self.best_intent_acc = intent_acc
        if slot_f1 > self.best_slot_f1:
            self.best_slot_f1 = slot_f1
            self.no_improvement_num = 0
            # only when best slot_f1 is found, we save the model
            self.save_model()
        else:
            if istestset == False:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
        
        if self.no_improvement_num >= self.early_stop:
            self.stop_training_flag = True
        
        return intent_acc, slot_f1, self.stop_training_flag

    def save_model(self):
        """
        save the best model (achieve best f1 on slot prediction)
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "text_encoder": self.lstm,
            "intent_predictor": self.intent_predictor,
            "slot_predictor": self.slot_predictor
        }, saved_path)
        
        logger.info("Best model has been saved to %s" % saved_path)
