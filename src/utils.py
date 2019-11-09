
import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta

import numpy as np
from tqdm import tqdm

def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)

def load_embedding(vocab, emb_dim, emb_file):
    logger = logging.getLogger()
    embedding = np.zeros((vocab.n_words, emb_dim))
    logger.info("embedding: %d x %d" % (vocab.n_words, emb_dim))
    assert emb_file is not None
    with open(emb_file, "r") as ef:
        logger.info('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        embedded_words = []
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in vocab.word2index and sp[0] not in embedded_words:
                pre_trained += 1
                embedding[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])
        logger.info("Pre-train: %d / %d (%.2f)" % (pre_trained, vocab.n_words, pre_trained / vocab.n_words))
        unembedded_words = [word for word in vocab.word2index if word not in embedded_words]
        # print(unembedded_words)

    return embedding

def add_special_embedding(emb_path1, emb_path2, emb_path3):
    special_token_list = ["<TIME>", "<LAST>", "<DATE>", "<LOCATION>", "<NUMBER>"]
    spe_tok_emb_dict = {}
    emb_dim = 300
    for spe_tok in special_token_list:
        random_emb = np.random.normal(0, 0.1, emb_dim).tolist()
        random_emb = [str(float_num) for float_num in random_emb]
        random_emb_str = " ".join(random_emb)
        spe_tok_emb_dict[spe_tok] = random_emb_str

    with open(emb_path1, "a+") as f:
        for spe_tok in tqdm(special_token_list):
            random_emb_str = spe_tok_emb_dict[spe_tok]
            f.write(spe_tok + " " + random_emb_str + "\n")
    with open(emb_path2, "a+") as f:
        for spe_tok in tqdm(special_token_list):
            random_emb_str = spe_tok_emb_dict[spe_tok]
            f.write(spe_tok + " " + random_emb_str + "\n")
    with open(emb_path3, "a+") as f:
        for spe_tok in tqdm(special_token_list):
            random_emb_str = spe_tok_emb_dict[spe_tok]
            f.write(spe_tok + " " + random_emb_str + "\n")

def write_seed_dict(src_lang, tgt_lang, out_path):
    
    seed_words_en = ["weather", "forecast", "temperature", "rain", "hot", "cold", "remind", "forget", "alarm", "cancel", "tomorrow"]
    seed_words_es = ["clima", "pronóstico", "temperatura", "lluvia", "caliente", "frío", "recordar", "olvidar", "alarma", "cancelar", "mañana"]
    seed_words_th = ["อากาศ", "พยากรณ์", "อุณหภูมิ", "ฝน", "ร้อน", "หนาว", "เตือน", "ลืม", "เตือน", "ยกเลิก", "พรุ่ง"]

    assert len(seed_words_en) == len(seed_words_es) == len(seed_words_th)

    seed_dict = {"en": seed_words_en, "es": seed_words_es, "th":seed_words_th}

    with open("../refine_emb/"+src_lang+"2"+tgt_lang+"_dict_final.pkl", "rb") as f:
        dictionary = pickle.load(f)
    src_words = list(dictionary.keys())
    tgt_words = list(dictionary.values())

    with open("../refine_emb/"+src_lang+"_wordlist.pkl", "rb") as f:
        task_src_words = pickle.load(f)
    with open("../refine_emb/"+tgt_lang+"_wordlist.pkl", "rb") as f:
        task_tgt_words = pickle.load(f)

    src_match_words = []
    for src_w in task_src_words:
        if src_w in src_words:
            src_match_words.append(src_w)
    print("%d / %d (%.4f)" % (len(src_match_words), len(task_src_words), len(src_match_words)*1.0/len(task_src_words)))

    new_dict = {}
    tgt_match_words = []
    for src_match_w in src_match_words:
        tgt_w = dictionary[src_match_w]
        if tgt_w in task_tgt_words:
            tgt_match_words.append(tgt_w)
            new_dict[src_match_w] = tgt_w
    print("%d / %d (%.4f)" % (len(tgt_match_words), len(src_match_words), len(tgt_match_words)*1.0/len(src_match_words)))

    src_words = set(list(new_dict.keys()))
    tgt_words = set(list(new_dict.values()))
    assert len(src_words) == len(tgt_words)

    seed_words_src = seed_dict[src_lang]
    seed_words_tgt = seed_dict[tgt_lang]
    for src_w, tgt_w in zip(src_words, tgt_words):
        # print(src_w, tgt_w)
        if src_w not in seed_words_src and tgt_w not in seed_words_tgt:
            seed_words_src.append(src_w)
            seed_words_tgt.append(tgt_w)
    
    assert len(seed_words_src) == len(seed_words_tgt)

    f = open(out_path, "w")
    for seed_w_src, seed_w_tgt in zip(seed_words_src, seed_words_tgt):
        f.write(seed_w_src + " " + seed_w_tgt + "\n")
    f.close()
