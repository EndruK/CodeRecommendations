from Helper.logging import init_logging
from Seq2Seq_Pytorch_test.Data.preprocessor import Preprocessor
import argparse

if __name__ == "__main__":
    """
    NOTE: this function currently only works with the pytorch version (Seq2Seq_Pytorch_test)
    """
    # TODO: what should be put into a config file?
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("dataset_path", type=str, help="Absolute path to serialized AST of Java methods")
    parser.add_argument("target_path", type=str, help="Absolute path to store the resulting csv file")
    parser.add_argument("log_level", type=str, help="Define the log level (debug, info, warn, critical)")
    parser.add_argument("log_path", type=str, help="Define the path to store the log file to")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    target_path = args.target_path
    log_level = args.log_level
    log_path = args.log_path
    # init logger
    init_logging(log_level, log_path)
    # run preprocessing
    Preprocessor.preprocess(dataset_path, target_path)
