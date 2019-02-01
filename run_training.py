from Helper.logging import init_logging
from Seq2Seq_Pytorch_test.Data.dataset import Dataset
from Seq2Seq_Pytorch_test.Model.vanilla_seq2seq import VanillaSeq2Seq
import argparse
import torch.utils.data as data


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

    # TODO: parameterize this
    hidden_size = 512
    batch_size = 32
    vocab_size = len(dataset.vocab)
    embedding_dimension = 256
    cuda_enabled = True
    epochs = 50
    validate_every_batch = 2000

    # build pytorch model
    model = VanillaSeq2Seq(
        hidden_size=hidden_size,
        batch_size=batch_size,
        vocab_size=vocab_size,
        embedding_dimension=embedding_dimension,
        cuda_enabled=cuda_enabled,
        sos_index=dataset.word_2_index["SOS"]
    )
    data_loader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 6
    }
    # TODO: override the collate_fn of DataLoader to PAD to longest element - and tokenize
    training_generator = data.DataLoader(dataset.partitions["training"], **data_loader_params)
    validation_generator = data.DataLoader(dataset.partitions["validation"], **data_loader_params)
    testing_generator = data.DataLoader(dataset.partitions["testing"], **data_loader_params)
    global_step = 0
    for i in range(epochs):
        # iterate over all items in training and bundle mini_batches in random order
        for batch_x, batch_y in training_generator:
            loss, acc = model.training_iteration(batch_x, batch_y, batch_y_mask)
            if global_step % print_every_batch == 0 and global_step > 0:
                # TODO: print here
                pass
            if global_step % validate_every_batch == 0 and global_step > 0:
                # TODO: validate here
                # TODO: save on improvement
                pass
            global_step += 1
    for batch_x, batch_y in testing_generator:
        loss, acc = model.validation_iteration(batch_x, batch_y, batch_y_mask)
        # TODO: print the result
