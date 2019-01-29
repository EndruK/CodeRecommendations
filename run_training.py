from Helper.logging import init_logging
from Seq2Seq_Pytorch_test.Data.dataset import Dataset
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("csv_path", type=str, help="Path to the CSV holding all tuples")
    parser.add_argument("tokenizer", type=str, help="tokenizer to use on training (nltk, json)")

    parser.add_argument("vocab_top_k", type=int, help="Amount of top k words to keep in vocab")
    parser.add_argument("vocab_build_processes", type=int, help="num of parallel processes for vocab creation")

    parser.add_argument("vocab_export_path", type=str, help="Path to dump the extracted vocab to")
    parser.add_argument("vocab_name", type=str, help="name of the resulting vocab files")

    parser.add_argument("log_level", type=str, help="Define the log level (debug, info, warn, critical)")
    parser.add_argument("log_path", type=str, help="Define the path to store the log file to")
    args = parser.parse_args()
    csv_path = args.csv_path
    tokenizer = args.tokenizer

    top_k = args.vocab_top_k
    num_processes = args.vocab_build_processes

    vocab_path = args.vocab_export_path
    vocab_name = args.vocab_name

    log_level = args.log_level
    log_path = args.log_path
    init_logging(log_level, log_path)

    if tokenizer == "nltk":
        from Seq2Seq_Pytorch_test.Data.tokenizers.nltk_tokenizer import NLTKTokenizer as Tok
    elif tokenizer == "json":
        from Seq2Seq_Pytorch_test.Data.tokenizers.json_tokenizer import JsonTokenizer as Tok
    else:
        raise ValueError("wrong tokenizer specified: %s" % tokenizer)
    # initialize the dataset
    dataset = Dataset(csv_path, Tok)
    dataset.split_dataset()
    dataset.build_vocab(top_k=top_k, num_processes=num_processes)
    dataset.dump_vocab(vocab_path, vocab_name)
    # now we can use the vocab for training

    # build pytorch model
    # run training on model