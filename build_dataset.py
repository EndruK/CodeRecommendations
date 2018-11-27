import logging as log, configparser, argparse
from Dataset.corpus import Corpus
from Dataset.json_processor import build_dataset


if __name__ == "__main__":
    log.basicConfig(
        level=log.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            log.StreamHandler()
        ])

    parser = argparse.ArgumentParser(description="Tool for building datasets for usage in a complete neural network" + \
                                                 "environment")
    parser.add_argument("config",
                        type=str,
                        help="the path where dataset_preprocessing config file is")
    parser.add_argument("destination",
                        type=str,
                        help="the path where the new dataset should be stored")
    args = parser.parse_args()
    dataset_preprocessing_config = configparser.RawConfigParser()
    dataset_preprocessing_config.read(args.config)

    corpus_path = dataset_preprocessing_config.get("Corpus", "path")
    num_of_processes = dataset_preprocessing_config.getint("Corpus", "processes")
    statement_limit = dataset_preprocessing_config.getint("Corpus", "statement_limit")
    file_max_limit = dataset_preprocessing_config.getint("Corpus", "file_max_limit")


    corpus = Corpus(
        path_to_corpus=corpus_path,
        partition_scheme=None,
        shuffle_files=True
    )

    build_dataset(
        corpus=corpus,
        process_count=num_of_processes,
        destination=args.destination,
        statement_limit=statement_limit,
        file_max_limit=file_max_limit
    )