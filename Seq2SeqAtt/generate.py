from Seq2SeqAtt.Generate.seq2seqAtt_generate import Generate
from Seq2SeqAtt.Utils.regex_tokenizer import RegexTokenizer
from Seq2SeqAtt.Preprocessing.Embedding.json_embedding import JSONEmbedding
from Seq2SeqAtt.Preprocessing.Dataset.seq2seq_dataset import Seq2SeqDataset
import argparse, configparser, sys, os, datetime, operator
def run():
    parser = argparse.ArgumentParser(description="Load configs")
    parser.add_argument('machine_config',
                        type=str,
                        help='path to the config for the executing machine')
    parser.add_argument('output_config',
                        type=str,
                        help='path to the config for the output details')
    parser.add_argument('experiment_config',
                        type=str,
                        help='path to the config for the current experiment')
    parser.add_argument('--experiment_path',
                        type=str,
                        help='path to where to store everything in',
                        default=".")
    args = parser.parse_args()

    machine_config = configparser.RawConfigParser()
    machine_config.read(args.machine_config)

    output_config = configparser.RawConfigParser()
    output_config.read(args.output_config)

    experiment_config = configparser.RawConfigParser()
    experiment_config.read(args.experiment_config)

    tokenizer = RegexTokenizer()
    vocab_path = os.path.join(args.experiment_path, output_config.get("Dumping", "vocab_path"))
    print("vocab", vocab_path)
    subset_paths = machine_config.get("Dataset", "subset_paths").split(",")
    subset_paths = [os.path.join(args.experiment_path, p) for p in subset_paths]

    dataset = Seq2SeqDataset(dataset_path=machine_config.get("Dataset", "corpus_path"),
                             tokenizer=tokenizer,
                             vocab_path=vocab_path,
                             dump_path=args.experiment_path,
                             subset_paths=subset_paths)
    dataset.load()

    embedding_log_path = os.path.join(
        args.experiment_path, output_config.get("Logging", "embedding_log_path"))
    embedding_checkpoint_path = os.path.join(
        args.experiment_path, output_config.get("Dumping", "embedding_checkpoint_path"))
    embedding_matrix_path = os.path.join(
        args.experiment_path, output_config.get("Dumping", "embedding_matrix_path"))
    embedding_model = JSONEmbedding(batch_size=experiment_config.getint("Embeddings", "batch_size"),
                                            dataset=dataset,
                                            embedding_size=experiment_config.getint("Embeddings", "hidden_size"),
                                            logs_path=embedding_log_path,
                                            model_checkpoint_path=embedding_checkpoint_path,
                                            epochs=experiment_config.getint("Embeddings", "epochs"),
                                            gpu=experiment_config.getint("Meta", "gpu"))

    # if os.path.isfile(embedding_matrix_path):
    #    print("restore embeddings from", embedding_matrix_path)
    #    embedding_model.restore_np_embeddings(embedding_matrix_path)
    # else:
    #    sys.exit(0)
    enc_hidden = experiment_config.getint("Model", "hidden_size")
    dec_hidden = experiment_config.getint("Model", "hidden_size")
    gen = Generate(dataset, embedding_model, enc_hidden, enc_hidden*2)

    training_checkpoint_path = os.path.join(
        args.experiment_path, output_config.get("Dumping", "training_checkpoint_path"))

    training_checkpoint_path = os.path.join(training_checkpoint_path, "best.checkpoint")

    # 2018-09-05 15:38:16.152787
    # get files in dir
    # files = os.listdir(training_checkpoint_path)
    # dates = {}
    # for file in files:
    #     lhs = file.split(".")[0]
    #     try:
    #         t = datetime.datetime.strptime(lhs, '%Y-%m-%d %H:%M:%S')
    #         if t not in dates and lhs != "checkpoint":
    #             dates[t] = file
    #     except:
    #         continue
    # sorted_dates = sorted([[key, value] for key,value in dates.items()], key=lambda x: x[1], reverse=True)
    # print(sorted_dates)
    #training_checkpoint_path = os.path.join(training_checkpoint_path, sorted_dates[0][1])

    gen.restore_checkpoint(training_checkpoint_path)
    gen.gen()