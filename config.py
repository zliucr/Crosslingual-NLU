import argparse

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Multilingual Task-Oriented Dialog")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="multilingual_dialogue.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--emb_file_en", type=str, default="./emb/wiki.en.align.vec", help="Path of word embeddings for English")
    parser.add_argument("--emb_file_es", type=str, default="./emb/wiki.es.align.vec", help="Path of word embeddings for Spanish")
    parser.add_argument("--emb_file_th", type=str, default="./emb/wiki.th.align.vec", help="Path of word embeddings for Thai")

    # model parameters
    parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for lstm")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of lstm layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=250, help="Hidden layer dimension")
    parser.add_argument("--freeze_emb", default=False, action="store_true", help="Freeze embedding or not")
    
    # use lvm
    parser.add_argument("--lvm", default=False, action="store_true", help="Use lvm")
    parser.add_argument("--lvm_dim", type=int, default=100, help="lvm dimension")

    # use crf
    parser.add_argument("--crf", default=False, action="store_true", help="Use CRF")

    # use emb noise
    parser.add_argument("--embnoise", default=False, action="store_true", help="Use embedding noise")

    # use special token replacement
    parser.add_argument("--clean_txt", default=False, action="store_true", help="clean text")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--early_stop", type=int, default=3, help="No improvement after several epoch, we stop training")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # transfer parameters
    parser.add_argument("--transfer", default=False, action="store_true", help="Tranfer to other language (Zero-shot)")
    parser.add_argument("--trans_lang", type=str, default="es", help="Choose a language to transfer (es, th)")

    # data statistic
    parser.add_argument("--num_intent", type=int, default=12, help="Number of intent in the dataset")
    parser.add_argument("--num_slot", type=int, default=24, help="Number of slot in the dataset")

    params = parser.parse_args()

    return params
