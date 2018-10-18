from Seq2Seq.Preprocessing.Dataset.seq2seq_dataset import Seq2SeqDataset
from Seq2Seq.Preprocessing.Dataset.JSON.dataset import JsonDataset
from Seq2Seq.Preprocessing.Dataset.seq2seq_sample_data import Seq2SeqSampleData
from Seq2Seq.Utils.regex_tokenizer import RegexTokenizer
from Seq2Seq.Preprocessing.Embedding.json_embedding import JSONEmbedding
from Seq2Seq.Model.seq2seq_train import Train
import argparse, configparser, os, sys, math, multiprocessing
import numpy as np

def seq2seq_generate():
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
    parser.add_argument('--input',
                        type=str,
                        help='x string')
    args = parser.parse_args()
    ####################################################################################################################
    ### Read Configs ###################################################################################################
    ####################################################################################################################

    machine_config = configparser.RawConfigParser()
    machine_config.read(args.machine_config)

    output_config = configparser.RawConfigParser()
    output_config.read(args.output_config)

    experiment_config = configparser.RawConfigParser()
    experiment_config.read(args.experiment_config)

    destination = os.path.join(args.experiment_path, "pre_built_dataset")
    dataset = JsonDataset(dataset_path=machine_config.get("Dataset", "corpus_path"),
                          output_path=args.experiment_path,
                          dump_path=destination)
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
    # embedding_model.restore_np_embeddings(embedding_matrix_path)

    training_log_path = os.path.join(
        args.experiment_path, output_config.get("Logging", "training_log_path"))
    training_checkpoint_path = os.path.join(
        args.experiment_path, output_config.get("Dumping", "training_checkpoint_path"))
    nn_model = Train(datamodel=dataset,
                     embedding_model=embedding_model,
                     logs_path=training_log_path,
                     epochs=experiment_config.getint("Model", "epochs"),
                     hidden_size=experiment_config.getint("Model", "hidden_size"),
                     learning_rate=experiment_config.getfloat("Model", "learning_rate"),
                     validation_interval=experiment_config.getint("Metric", "validation_interval"),
                     checkpoint_path=training_checkpoint_path,
                     gpu=experiment_config.getint("Meta", "gpu"),
                     print_interval=experiment_config.getint("Metric", "print_interval"),
                     batch_size=experiment_config.getint("Model", "batch_size"))

    if len(args.input) < 1:
        test_batch_generator = dataset.pre_build_pair_batch_generator("testing", batch_size=16)
        input_xym = next(test_batch_generator)
    else:
        tokens = dataset.indexize_text(args.input)
        tokens = dataset.pad(tokens, dataset.input_size)
        # TODO: this is not optimal --- tf expects the batch size which it used to train on
        batch = [tokens] * experiment_config.getint("Model", "batch_size")
        x = batch
        y = np.zeros([experiment_config.getint("Model", "batch_size"), dataset.output_size])
        mask = np.zeros([experiment_config.getint("Model", "batch_size"), dataset.output_size])
        input_xym = (x, y, mask)
    result = nn_model.generate(input_xym)
    print(result)
