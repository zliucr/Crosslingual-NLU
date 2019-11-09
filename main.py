
from config import get_params
from src.utils import init_experiment, load_embedding
from src.loader import get_dataloader
from src.lstm import Lstm, IntentPredictor, SlotPredictor
from src.trainer import DialogTrainer
from src.transfer import EvaluateTransfer

import numpy as np
from tqdm import tqdm

import pickle
import os

def train(params, lang="en"):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    # dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params, lang=lang)

    # build model
    lstm = Lstm(params, vocab)
    intent_predictor = IntentPredictor(params)
    slot_predictor = SlotPredictor(params)
    lstm.cuda()
    intent_predictor.cuda()
    slot_predictor.cuda()

    # build trainer
    dialog_trainer = DialogTrainer(params, lstm, intent_predictor, slot_predictor)
    
    for e in range(params.epoch):
        logger.info("============== epoch %d ==============" % e)
        intent_loss_list, slot_loss_list = [], []
        
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        for i, (X, lengths, y1, y2) in pbar:
            X, lengths, y1 = X.cuda(), lengths.cuda(), y1.cuda()  # the length of y2 is different for each sequence
            intent_loss, slot_loss = dialog_trainer.train_step(e, X, lengths, y1, y2)
            intent_loss_list.append(intent_loss)
            slot_loss_list.append(slot_loss)
            
            pbar.set_description("(Epoch {}) INTENT LOSS:{:.4f} SLOT LOSS:{:.4f}".format(e, np.mean(intent_loss_list), np.mean(slot_loss_list)))

        logger.info("Finish training epoch %d. Intent loss: %.4f. Slot loss: %.4f" % (e, np.mean(intent_loss_list), np.mean(slot_loss_list)))
        
        logger.info("============== Evaluate %d ==============" % e)

        intent_acc, slot_f1, stop_training_flag = dialog_trainer.evaluate(dataloader_val)
        logger.info("Intent ACC: %.4f (Best Acc: %.4f). Slot F1: %.4f. (Best F1: %.4f)" % (intent_acc, dialog_trainer.best_intent_acc, slot_f1, dialog_trainer.best_slot_f1))

        if stop_training_flag == True:
            break

    logger.info("============== Final Test ==============")
    intent_acc, slot_f1, _ = dialog_trainer.evaluate(dataloader_test, istestset=True)
    logger.info("Intent ACC: %.4f. Slot F1: %.4f." % (intent_acc, slot_f1))

def transfer(params, trans_lang):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info("============== Evaluate Zero-Shot on %s ==============" % trans_lang)

    # dataloader
    _, _, dataloader_test, vocab = get_dataloader(params, lang=trans_lang)
    
    # get word embedding
    emb_file = params.emb_file_es if trans_lang == "es" else params.emb_file_th
    embedding = load_embedding(vocab, params.emb_dim, emb_file)
    
    # evaluate zero-shot
    evaluate_transfer = EvaluateTransfer(params, dataloader_test, embedding, vocab.n_words)
    intent_acc, slot_f1 = evaluate_transfer.evaluate()
    logger.info("Intent ACC: %.4f. Slot F1: %.4f." % (intent_acc, slot_f1))

if __name__ == "__main__":
    params = get_params()
    if params.transfer == False:
        train(params, lang="en")
    else:
        transfer(params, params.trans_lang)
    