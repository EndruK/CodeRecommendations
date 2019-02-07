import pandas as pd
import logging as log
from sklearn.model_selection import train_test_split
import time
import datetime
from Seq2Seq_Pytorch_test.Data.partition import Partition
from Seq2Seq_Pytorch_test.Data.tokenizers.json_tokenizer import JsonTokenizer
from typing import Dict
import multiprocessing
import math
import os
import pickle as pkl


class Dataset:

    SPECIAL_TOKENS = ["UNK", "PAD", "EMPTY", "INV", "SOS", "EOS"]

    def __init__(self, dataset_path, tokenizer):
        """
        Init the dataset - nothing very special.
        :param dataset_path: path to the csv file holding all tuples of preprocessor
        :param tokenizer: class reference to a tokenizer
        """
        self.dataset_path = dataset_path
        self.partitions = {"training": None, "validation": None, "testing": None}  # type: Dict[str, Partition]
        self.tokenizer = tokenizer
        self.vocab = []
        self.index_2_word = {}
        self.word_2_index = {}

    def __load_csv(self):
        """
        Load the csv file located at self.dataset_path which was given in the init function.
        :return: pandas dataframe holding all tuples of the preprocessing
        """
        log.debug("start loading csv file into RAM")
        _dataset = pd.read_csv(self.dataset_path, delimiter=",", lineterminator="\n", quotechar="'")
        log.debug("done loading csv file")
        log.info("tuples in dataset: %d" % len(_dataset))
        return _dataset

    def split_dataset(self):
        """
        Split the dataset into training, validation and testing - filling the member-attribute partitions.
        """
        log.debug("splitting dataset")

        # load csv file to RAM
        _dataset = self.__load_csv()
        # split dataframe into sources and targets (x and y are symmetric)
        _x = _dataset.iloc[:, 0].values
        _y = _dataset.iloc[:, 1].values

        # first split to get training parts
        _train_x, _validation_and_test_x, _train_y, _validation_and_test_y = train_test_split(_x, _y,
                                                                                              test_size=.2,
                                                                                              random_state=3,
                                                                                              shuffle=True)
        # second split to get validation and testing parts
        _validation_x, _test_x, _validation_y, _test_y = train_test_split(_validation_and_test_x,
                                                                          _validation_and_test_y,
                                                                          test_size=.4,
                                                                          random_state=42,
                                                                          shuffle=True)
        # create pytorch datasets as partitions
        self.partitions["training"] = Partition(_train_x, _train_y, self.tokenizer)
        self.partitions["validation"] = Partition(_validation_x, _validation_y, self.tokenizer)
        self.partitions["testing"] = Partition(_test_x, _test_y, self.tokenizer)
        log.info("sizes of the dataset:")
        log.info("training: %d" % len(self.partitions["training"]))
        log.info("validation: %d" % len(self.partitions["validation"]))
        log.info("testing: %d" % len(self.partitions["testing"]))

    def build_vocab(self, top_k, num_processes=5, include_y=False):
        """
        Multi-Process version. Works on small CSV files but has problems with larger.
        Connected issue: https://github.com/EndruK/CodeRecommendations/issues/1
        Create the vocabulary on the training dataset.x strings and the tokenizer of this dataset
        also, fills the member-attributes of vocab, index_2_word and word_2_index.
        Alternatively, you can just load a vocab dump using "load_vocab" - but keep in mind that the
        vocab is based on a temporary partition scheme!
        :param top_k: number of top-k tokens in the resulting vocabulary
        :param num_processes: how many processes should be used to create the vocabulary (default=5)
        :param include_y: flag whether to include tokens of target y to vocab or not (default=False)
        """
        # assume there is something in the training partition
        assert self.partitions["training"] is not None
        _vocab = {}
        log.debug("start vocab creation - multi process")
        log.debug("#processes: %d" % num_processes)
        processes = []
        # spawn a process pool
        pool = multiprocessing.Pool(processes=num_processes)
        _start = time.time()
        # how many sources should a process cover?
        items_per_process = math.floor(len(self.partitions["training"]) / num_processes)
        # create a distribution for all processes
        distribution = [items_per_process] * num_processes
        # add the remaining indices to the distribution
        remainder = len(self.partitions["training"]) % num_processes
        for i in range(remainder):
            distribution[i] += 1

        # first start index is 0
        start_index = 0
        end_index = 0
        for process_id in range(num_processes):
            # add current distribution value to end index to get the range inside the partition
            end_index += distribution[process_id]
            # spawn a new process and add it to the process list
            process = pool.apply_async(Dataset.build_vocab_parallel,
                                       args=(process_id,
                                             self.partitions["training"],
                                             self.tokenizer,
                                             start_index,
                                             end_index,
                                             include_y))
            processes.append(process)
            # start index is always last end index
            start_index = end_index
        for process in processes:
            # wait for all processes to terminate
            result = process.get()
            # merge vocabs
            for key, value in result.items():
                if key not in _vocab:
                    _vocab[key] = value
                else:
                    _vocab[key] += value
        _end = time.time()
        log.debug("complete vocab creation time: %s" % datetime.timedelta(seconds=_end-_start))
        # sort vocab based on token frequency
        _sorted_vocab = sorted([[key, value] for key, value in _vocab.items()],
                               key=lambda j: j[1],
                               reverse=True)
        log.info("size of vocab before removing of non top-k elements: %d" % len(_vocab))
        log.debug("top 10 elements in vocab: %s" % str(_sorted_vocab[:10]))
        log.debug("adding special tokens: %s" % str(Dataset.SPECIAL_TOKENS))
        # keep top-k tokens as vocab
        _sorted_vocab = _sorted_vocab[:top_k]
        vocab_list = [token for token, _ in _sorted_vocab]
        # add special tokens to the left of the vocab
        vocab_list = Dataset.SPECIAL_TOKENS + vocab_list

        # create index mappings
        self.vocab = vocab_list
        self.index_2_word = {}
        self.word_2_index = {}
        for i in range(len(self.vocab)):
            word = self.vocab[i]
            self.index_2_word[i] = word
            self.word_2_index[word] = i

        self.partitions["training"].set_vocab_and_mapping(self.vocab, self.word_2_index)
        self.partitions["validation"].set_vocab_and_mapping(self.vocab, self.word_2_index)
        self.partitions["testing"].set_vocab_and_mapping(self.vocab, self.word_2_index)

    def build_vocab_single_process(self, top_k, include_y=False):
        """
        Main-Process version. Workaround for https://github.com/EndruK/CodeRecommendations/issues/1
        Create the vocabulary on the training dataset.x strings and the tokenizer of this dataset
        also, fills the member-attributes of vocab, index_2_word and word_2_index.
        Alternatively, you can just load a vocab dump using "load_vocab" - but keep in mind that the
        vocab is based on a temporary partition scheme!
        :param top_k: number of top-k tokens in the resulting vocabulary
        :param num_processes: how many processes should be used to create the vocabulary (default=5)
        :param include_y: flag whether to include tokens of target y to vocab or not (default=False)
        """
        # assume there is something in the training partition
        assert self.partitions["training"] is not None
        _vocab = {}
        log.debug("start vocab creation - single process")
        _start = time.time()
        t = self.tokenizer()
        cnt = 1
        for x, y in self.partitions["training"]:
            if cnt % 200 == 0:
                log.debug("process tuple %d of %d" % (cnt, len(self.partitions["training"])))
            tokens = t.tokenize(x)
            for token in tokens:
                if token not in _vocab:
                    _vocab[token] = 1
                else:
                    _vocab[token] += 1
            if include_y:
                tokens = t.tokenize(y)
                for token in tokens:
                    if token not in _vocab:
                        _vocab[token] = 1
                    else:
                        _vocab[token] += 1
            cnt += 1
        _end = time.time()
        log.debug("complete vocab creation time: %s" % datetime.timedelta(seconds=_end - _start))
        # sort vocab based on token frequency
        _sorted_vocab = sorted([[key, value] for key, value in _vocab.items()],
                               key=lambda j: j[1],
                               reverse=True)
        log.info("size of vocab before removing of non top-k elements: %d" % len(_vocab))
        log.debug("top 10 elements in vocab: %s" % str(_sorted_vocab[:10]))
        log.debug("adding special tokens: %s" % str(Dataset.SPECIAL_TOKENS))
        # keep top-k tokens as vocab
        _sorted_vocab = _sorted_vocab[:top_k]
        vocab_list = [token for token, _ in _sorted_vocab]
        # add special tokens to the left of the vocab
        vocab_list = Dataset.SPECIAL_TOKENS + vocab_list

        # create index mappings
        self.vocab = vocab_list
        self.index_2_word = {}
        self.word_2_index = {}
        for i in range(len(self.vocab)):
            word = self.vocab[i]
            self.index_2_word[i] = word
            self.word_2_index[word] = i

        self.partitions["training"].set_vocab_and_mapping(self.vocab, self.word_2_index)
        self.partitions["validation"].set_vocab_and_mapping(self.vocab, self.word_2_index)
        self.partitions["testing"].set_vocab_and_mapping(self.vocab, self.word_2_index)

    def dump_vocab(self, p, title):
        """
        Dump the vocab, i2w and w2i variables to disk.
        :param p: path on the disk to store variables to
        :param title: name of the files
        """
        # assume we have a vocab and index mappings
        assert len(self.vocab) is not 0
        assert len(self.word_2_index) is not 0
        assert len(self.index_2_word) is not 0
        v_path = os.path.join(p, title + ".vocab")
        i2w_path = os.path.join(p, title + ".i2w")
        w2i_path = os.path.join(p, title + ".w2i")
        # check if there are already files and delete them if yes
        Dataset.check_and_delete(v_path)
        Dataset.check_and_delete(i2w_path)
        Dataset.check_and_delete(w2i_path)
        # dump variables to files
        with open(v_path, "wb") as f:
            pkl.dump(self.vocab, f)
        with open(i2w_path, "wb") as f:
            pkl.dump(self.index_2_word, f)
        with open(w2i_path, "wb") as f:
            pkl.dump(self.word_2_index, f)
        log.debug("done dumping vocab and indexing dicts to %s under the title %s" % (p, title))

    def load_vocab(self, p, title):
        """
        Load the vocab, i2w and w2i at the given path from disk.
        :param p: path on the disk where variables are stored
        :param title: name of the dumps
        """
        v_path = os.path.join(p, title + ".vocab")
        i2w_path = os.path.join(p, title + ".i2w")
        w2i_path = os.path.join(p, title + ".w2i")
        # assume there are files
        assert os.path.isfile(v_path)
        assert os.path.isfile(i2w_path)
        assert os.path.isfile(w2i_path)
        with open(v_path, "rb") as f:
            self.vocab = pkl.load(f)
        with open(i2w_path, "rb") as f:
            self.index_2_word = pkl.load(f)
        with open(w2i_path, "rb") as f:
            self.word_2_index = pkl.load(f)
        log.debug("loaded pickled vocab, i2w and w2i files at %s with title %s" % (p, title))

    def text_to_index_array(self, text):
        """
        Tokenize a given text based on current vocabulary and current tokenizer and transform it into
        word index representation.
        Also, adds start of sequence to left and end of sequence to right.
        :param text: the text
        :return: array, tokenized text in index format
        """
        tokenizer = self.tokenizer()
        tokens = tokenizer.tokenize(text)
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.word_2_index[token])
            else:
                result.append(self.word_2_index["UNK"])
        result = result + [self.index_2_word["EOS"]]
        return result

    @staticmethod
    def check_and_delete(p):
        """
        Check if there is a file at the given path and remove it
        :param p: path to a file
        """
        if os.path.isfile(p):
            log.debug("found file at target - removing %s" % p)
            os.remove(p)

    @staticmethod
    def build_vocab_parallel(process_id, partition, tokenizer, start, end, include_y=False):
        """
        Method to be called on sub-processes to extract a vocabulary on the subset of the given partition.
        :param process_id: ID of the process
        :param partition: reference to the dataset partition
        :param tokenizer: use the given tokenizer to extract tokens of text
        :param start: start index in the partition
        :param end: end index in the partition
        :param include_y: flag whether to include tokens of target y to vocab or not (default=False)
        :return: vocab of the subset of the partition
        """
        _vocab = {}
        t = tokenizer()
        for i in range(start, end):
            relative_index = i - start
            if relative_index % 200 == 0:
                log.debug("process %d processing %d of %d" % (process_id, relative_index, end-start))
            x, y = partition[i]
            tokens = t.tokenize(x)
            for token in tokens:
                if token not in _vocab:
                    _vocab[token] = 1
                else:
                    _vocab[token] += 1
            if include_y:
                tokens = t.tokenize(y)
                for token in tokens:
                    if token not in _vocab:
                        _vocab[token] = 1
                    else:
                        _vocab[token] += 1
        return _vocab


if __name__ == "__main__":
    log.basicConfig(
        level=log.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            log.StreamHandler()
        ]
    )
    path = "/home/andre/Documents/tuples_subset.csv"
    vocab_path = "/home/andre/Documents"
    ds = Dataset(path, JsonTokenizer)
    ds.split_dataset()
    ds.build_vocab(top_k=10000, num_processes=6)
    ds.dump_vocab(vocab_path, "test")
