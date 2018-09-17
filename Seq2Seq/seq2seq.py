from Seq2Seq.Preprocessing.Dataset.seq2seq_dataset import Seq2SeqDataset
from Seq2Seq.Preprocessing.Dataset.JSON.dataset import JsonDataset
from Seq2Seq.Preprocessing.Dataset.seq2seq_sample_data import Seq2SeqSampleData
from Seq2Seq.Utils.regex_tokenizer import RegexTokenizer
from Seq2Seq.Preprocessing.Embedding.json_embedding import JSONEmbedding
from Seq2Seq.Model.seq2seq_train import Train
import argparse, configparser, os, sys

def run():
    ####################################################################################################################
    ### Parse Arguments ################################################################################################
    ####################################################################################################################

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
    ####################################################################################################################
    ### Read Configs ###################################################################################################
    ####################################################################################################################

    machine_config = configparser.RawConfigParser()
    machine_config.read(args.machine_config)

    output_config = configparser.RawConfigParser()
    output_config.read(args.output_config)

    experiment_config = configparser.RawConfigParser()
    experiment_config.read(args.experiment_config)

    ####################################################################################################################
    ### Create Dataset #################################################################################################
    ####################################################################################################################

    # use the regex tokenizer
    # tokenizer = RegexTokenizer()

    # join the given experiment_path
    # vocab_path = os.path.join(args.experiment_path, output_config.get("Dumping", "vocab_path"))

    # subset_paths = machine_config.get("Dataset", "subset_paths").split(",")
    # subset_paths = [os.path.join(args.experiment_path, p) for p in subset_paths]

    # create vocab
    # dataset = Seq2SeqDataset(dataset_path=machine_config.get("Dataset", "corpus_path"),
    #                          tokenizer=tokenizer,
    #                          vocab_path=vocab_path,
    #                          subset_paths=subset_paths)
    dataset = JsonDataset(dataset_path=machine_config.get("Dataset", "corpus_path"),
                          output_path=args.experiment_path)


    if experiment_config.getboolean("Meta", "preprocess_dataset"):
        # dataset.create(shuffle=experiment_config.getboolean("Model", "shuffle_dataset"))
        subset_path = "/media/andre/E896A5A496A573AA/Corpora/AndreKarge_2018-09-12_JavaAST_JSON_without_features/Preprocessed/Sliced_idx/subset_50000.pkl"
        dataset.create(shuffle=True, word_threshold=5, subset=subset_path)
    else:
        dataset.load()

    ####################################################################################################################
    ### Create Dataset Sampler #########################################################################################
    ####################################################################################################################

    # dataset_sampler = Seq2SeqSampleData(dataset=dataset,
    #                                     input_sample_size=experiment_config.getint("Model", "input_size"),
    #                                     output_sample_size=experiment_config.getint("Model", "output_size"),
    #                                     batch_size=experiment_config.getint("Model", "batch_size"))

    ####################################################################################################################
    ### Train Embeddings ###############################################################################################
    ####################################################################################################################
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
    if experiment_config.getboolean("Meta", "train_embeddings"):
        embedding_model.train()
        embedding_model.store_embedding_as_np(embedding_matrix_path)
    else:
        if os.path.isfile(embedding_matrix_path):
            print("restore embeddings from", embedding_matrix_path)
            embedding_model.restore_np_embeddings(embedding_matrix_path)
        else:
            print("try to restore trained embedding checkpoint")
            try:
                embedding_model.load_checkpoint(embedding_checkpoint_path)
                embedding_model.store_embedding_as_np(embedding_matrix_path)
            except:
                print("unable to restore or load embeddings - aborting")
                sys.exit(0)
    ####################################################################################################################
    ### Train Neural Network ###########################################################################################
    ####################################################################################################################
    # training_log_path = os.path.join(
    #     args.experiment_path, output_config.get("Logging", "training_log_path"))
    # training_checkpoint_path = os.path.join(
    #     args.experiment_path, output_config.get("Dumping", "training_checkpoint_path"))
    # nn_model = Train(datamodel=dataset,
    #                  sampler=dataset_sampler,
    #                  embedding_model=embedding_model,
    #                  logs_path=training_log_path,
    #                  epochs=experiment_config.getint("Model", "epochs"),
    #                  hidden_size=experiment_config.getint("Model", "hidden_size"),
    #                  learning_rate=experiment_config.getfloat("Model", "learning_rate"),
    #                  validation_interval=experiment_config.getint("Metric", "validation_interval"),
    #                  checkpoint_path=training_checkpoint_path,
    #                  gpu=experiment_config.getint("Meta", "gpu"),
    #                  print_interval=experiment_config.getint("Metric", "print_interval"))
    # nn_model.train()