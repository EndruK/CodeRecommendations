from Seq2Seq_Pytorch_test.Preprocessing.Dataset.JSON.dataset import JsonDataset
from Seq2Seq_Pytorch_test.Model.seq2seq_train_pytorch import run_training as training_vanilla
from Seq2Seq_Pytorch_test.Model.seq2seq_attention_train_pytorch import run_training as training_attention
import argparse, configparser, os, math, multiprocessing, logging


def process_files(dataset, file_list, name, processid):
    dataset.pre_build_dataset_pairs(file_list, name, processid)
    return True

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
    parser.add_argument('--log_level',
                        type=str,
                        help='log level (debug, info=default, warn, critical)',
                        default='info')
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
    ### Set up Logging #################################################################################################
    ####################################################################################################################

    logging_level_string = args.log_level
    if logging_level_string == "info":
        logging_level = logging.INFO
    elif logging_level_string == "debug":
        logging_level = logging.DEBUG
    elif logging_level_string == "warn":
        logging_level = logging.WARN
    elif logging_level_string == "critical":
        logging_level = logging.CRITICAL
    else:
        logging_level = logging.INFO
    log_file = os.path.join(args.experiment_path, "training.log")

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.debug("initialized logger")
    logging.info("Log Level = " + args.log_level)

    ####################################################################################################################
    ### Create Dataset #################################################################################################
    ####################################################################################################################
    destination = os.path.join(args.experiment_path, "pre_built_dataset")
    logging.debug("initializing dataset object")
    logging.debug("experiment path = " + args.experiment_path)
    logging.debug("dataset position of preprocessed training pairs = " + destination)
    dataset = JsonDataset(dataset_path=machine_config.get("Dataset", "corpus_path"),
                          output_path=args.experiment_path,
                          dump_path=destination)



    if experiment_config.getboolean("Meta", "preprocess_dataset"):
        dataset.create(shuffle=True, word_threshold=5)
    else:
        dataset.load()

    if not os.path.isdir(destination):
        os.makedirs(destination)
    if len(os.listdir(destination)) == 0:
        logging.debug("start preprocessing training tuples")

        def parallel(data, name):
            num_threads = 7  # TODO: magic number
            length = math.floor(len(data) / num_threads)
            jobs = []
            pool = multiprocessing.Pool(processes=num_threads)
            for i in range(num_threads):
                start_index = i * length
                end_index = (i+1) * length
                if i == num_threads-1:
                    process_file_list = data[start_index:]
                else:
                    process_file_list = data[start_index:end_index]
                # (dataset, file_list, name, processid)
                p = pool.apply_async(process_files, args=(dataset, process_file_list, name, i))
                jobs.append(p)
            bla = []
            for p in jobs:
                p.get()
        parallel(dataset.training_files, "training")
        parallel(dataset.validation_files, "validation")
        parallel(dataset.testing_files, "testing")
    else:
        logging.info("found preprocessed training tuples - skipping building tuples")
        logging.debug("preprocessed tuple path: " + destination)

    training_log_path = os.path.join(
        args.experiment_path, output_config.get("Logging", "training_log_path"))
    training_checkpoint_path = os.path.join(
        args.experiment_path, output_config.get("Dumping", "training_checkpoint_path"))

    # training_vanilla(
    #     encoder_time_size=experiment_config.getint("Model", "input_size"),
    #     decoder_time_size=experiment_config.getint("Model", "output_size"),
    #     hidden_size=experiment_config.getint("Model", "hidden_size"),
    #     vocab_len=len(dataset.vocab),
    #     embedding_dim=experiment_config.getint("Embeddings", "hidden_size"),
    #     batch_size=experiment_config.getint("Model", "batch_size"),
    #     learning_rate=experiment_config.getfloat("Model", "learning_rate"),
    #     sos=dataset.w2i[dataset.SOS],
    #     eos=dataset.w2i[dataset.EOS],
    #     epochs=experiment_config.getint("Model", "epochs"),
    #     validation_interval=experiment_config.getint("Metric", "validation_interval"),
    #     log_interval=experiment_config.getint("Metric", "print_interval"),
    #     data=dataset,
    #     use_cuda=True,
    #     tensorboard_log_dir=training_log_path,
    #     model_store_path=training_checkpoint_path
    # )
    training_attention(
        encoder_time_size=experiment_config.getint("Model", "input_size"),
        decoder_time_size=experiment_config.getint("Model", "output_size"),
        hidden_size=experiment_config.getint("Model", "hidden_size"),
        vocab_len=len(dataset.vocab),
        embedding_dim=experiment_config.getint("Embeddings", "hidden_size"),
        batch_size=experiment_config.getint("Model", "batch_size"),
        learning_rate=experiment_config.getfloat("Model", "learning_rate"),
        sos=dataset.w2i[dataset.SOS],
        eos=dataset.w2i[dataset.EOS],
        epochs=experiment_config.getint("Model", "epochs"),
        validation_interval=experiment_config.getint("Metric", "validation_interval"),
        log_interval=experiment_config.getint("Metric", "print_interval"),
        data=dataset,
        use_cuda=True,
        tensorboard_log_dir=training_log_path,
        model_store_path=training_checkpoint_path
    )