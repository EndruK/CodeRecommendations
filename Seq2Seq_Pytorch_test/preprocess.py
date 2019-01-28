from Seq2Seq_Pytorch_test.Data.preprocessor import Preprocessor
import argparse
import logging as log


if __name__ == "__main__":
    # TODO: make this configurable
    log.basicConfig(
        level=log.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            log.StreamHandler()
        ]
    )
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("dataset_path", type=str, help="Absolute path to serialized AST of Java methods")
    parser.add_argument("target_path", type=str, help="Absolute path to store the resulting csv file")
    args = parser.parse_args()

    path = args.dataset_path
    target = args.target_path

    log.debug("dataset_path: %s" % path)
    log.debug("target_path: %s" % target)

    Preprocessor.preprocess(args.dataset_path, args.target_path)
