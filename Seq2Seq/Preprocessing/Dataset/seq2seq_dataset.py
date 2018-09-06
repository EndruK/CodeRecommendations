import os, random, pickle as pkl
import Seq2Seq.Utils.helper as utils

class Seq2SeqDataset:
    """
    Preprocessor class of the old AST model
    when initialized, it reads the given dataset path and builds
    a path array for training, validation and testing.
    These individual path subsets will get shuffled if shuffling is enabled.
    Additionally, it will create a vocabulary using the given tokenizer
    with word occurrencies higher than the given threshold.
    The initialization of this class will create 4 files.
    - a files/vocab file containing the vocab dictionary as pickle file
    - a files/training_paths file containing the array of all full paths for training
    - a files/validation_paths file containing the array of all full paths for validation
    - a files/testing_paths file containing the array of all full paths for testing
    """

    def __init__(self,
                 dataset_path,
                 tokenizer,
                 vocab_path,
                 subset_paths):
        super(Seq2SeqDataset, self).__init__()
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        self.dataset_root_path = dataset_path
        self.tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.subset_paths = subset_paths

    def create(self,
               shuffle=False,
               word_threshold=5):
        self.word_threshold = word_threshold
        # retreive all files in dataset
        self.all_files = utils.get_fullpath(self.dataset_root_path)
        self.shuffle = shuffle
        self.create_dataset_subsets()
        self.vocab = self.build_vocab(self.train_samples)
        self.word_to_index = self.build_word_to_index_mapping(self.vocab)
        self.index_to_word = self.build_index_to_word_mapping(self.vocab)
        self.export_vocab(self.vocab)
        self.export_path_sets(self.train_samples,
                              self.validation_samples, self.test_samples)

    def load(self):
        with open(self.vocab_path, "rb") as f:
            self.vocab = pkl.load(f)
        self.word_to_index = self.build_word_to_index_mapping(self.vocab)
        self.index_to_word = self.build_index_to_word_mapping(self.vocab)
        with open(self.subset_paths[0], "rb") as f:
            self.train_samples = pkl.load(f)
        with open(self.subset_paths[1], "rb") as f:
            self.validation_samples = pkl.load(f)
        with open(self.subset_paths[2], "rb") as f:
            self.test_samples = pkl.load(f)
        print("Model loaded")

    def build_word_to_index_mapping(self, vocab):
        """
        index mapping word -> index
        """
        mapping = {}
        for index, word in enumerate(vocab):
            mapping[word] = index
        return mapping

    def build_index_to_word_mapping(self, vocab):
        """
        index mapping index -> word
        """
        mapping = {}
        for index, word in enumerate(vocab):
            mapping[index] = word
        return mapping

    def create_dataset_subsets(self):
        """
        create dataset subsets for training, validation, testing
        shuffle if necessary
        """
        # files in the old dataset are separated into train/valid/test
        # -> separate them into own lists
        train_files, valid_files, test_files = self.split_folders(
            self.all_files)
        # samples: [ast, methodParameter, methods, variables, activatedSwingElements]
        self.train_samples = self.split_files(train_files)
        self.validation_samples = self.split_files(valid_files)
        self.test_samples = self.split_files(test_files)

    def export_path_sets(self, train, validation, testing):
        """
        stores the sets for later usage on disk
        """
        self.export_path_set(train, self.subset_paths[0])
        self.export_path_set(validation, self.subset_paths[1])
        self.export_path_set(testing, self.subset_paths[2])

    def export_path_set(self, subset, path):
        """
        export a single subset
        """
        if os.path.isfile(path):
            # uin = input(path + " file aleady exists. Override? (y/n)")
            uin = ""
            if uin == "n" or uin == "N":
                return
            else:
                os.remove(path)
                print("old " + path + " file removed")
        with open(path, "wb") as f:
            pkl.dump(subset, f)

    def export_vocab(self, vocab):
        """
        stores a vocabulary on disk
        """
        # check file first
        if os.path.isfile(self.vocab_path):
            # uin = input("Vocab file aleady exists. Override? (y/n)")
            uin = ""
            if uin == "n" or uin == "N":
                return
            else:
                os.remove(self.vocab_path)
                print("old vocab file removed")
        with open(self.vocab_path, "wb") as f:
            pkl.dump(vocab, f)
        print("vocabulary stored at", self.vocab_path)

    def ast_file_handler(self, file):
        """
        handle an AST file
        """
        tokens = self.tokenizer.tokenize(
            file)
        return tokens

    def ast_result_handler(self, vocab, file_result):
        """
        handle the result of the ast_file_handler
        """
        if file_result not in vocab:
            vocab[file_result] = 1
        else:
            vocab[file_result] += 1
        return vocab

    def feature_file_handler(self, file):
        """
        handle any feature file
        """
        result = []
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            l = line.strip()
            if len(l) > 1:
                result.append(l)
        return result

    def feature_line_handler(self, vocab, line):
        """
        handle a line of a feature file
        """
        for part in line.split(","):
            if part not in vocab:
                vocab[part] = 1
            else:
                vocab[part] += 1
        return vocab

    def build_vocab(self, dataset):
        """
        build the vocabulary
        creates for each file type in dataset a vocab
        and joins them together afterwards
        at the end, all words with a low frequency get
        removed
        """
        ast_vocab = self.build_subset_vocab(dataset["ast.sliced"],
                                            self.ast_file_handler,
                                            self.ast_result_handler,
                                            name="ast_vocab")
        variable_vocab = self.build_subset_vocab(dataset["variables"],
                                                 self.feature_file_handler,
                                                 self.feature_line_handler,
                                                 name="variable_vocab")
        method_param_vocab = self.build_subset_vocab(dataset["methodParameter"],
                                                     self.feature_file_handler,
                                                     self.feature_line_handler,
                                                     name="method_param_vocab")
        method_vocab = self.build_subset_vocab(dataset["methods"],
                                               self.feature_file_handler,
                                               self.feature_line_handler,
                                               name="method_vocab")
        # swing can use the ast line handler since there are only identifier in each line
        swing_vocab = self.build_subset_vocab(dataset["activatedSwingElements"],
                                              self.feature_file_handler,
                                              self.ast_result_handler,
                                              name="swing_vocab")

        # join vocabs together
        vocab = {}
        vocab = utils.join_dictionaries(vocab, ast_vocab)
        vocab = utils.join_dictionaries(vocab, variable_vocab)
        vocab = utils.join_dictionaries(vocab, method_param_vocab)
        vocab = utils.join_dictionaries(vocab, method_vocab)
        vocab = utils.join_dictionaries(vocab, swing_vocab)

        vocab = self.remove_low_frequency_words(vocab, self.word_threshold)
        vocab["<UNK>"] = -1
        vocab["<INV>"] = -1
        vocab["<PAD>"] = -1
        vocab["<GO>"] = -1
        return vocab

    def build_subset_vocab(self, subset, file_handler, result_handler, name):
        """
        generic vocab builder
        file handler defines how files are processed
        result handler defines how the file result is processed
        """
        vocab = {}
        for subset_file in subset:
            file_results = file_handler(subset_file)
            for file_result in file_results:
                vocab = result_handler(vocab, file_result)
        print(name, "length:", len(vocab))
        return vocab

    def remove_low_frequency_words(self, vocab, threshold):
        """
        removes words from vocab with a low frequency
        """
        print("vocab length before cleansing:", len(vocab))
        remove_list = []
        for word, frequency in vocab.items():
            if frequency < threshold:
                remove_list.append(word)
        for word in remove_list:
            vocab.pop(word)
        print("vocab length after cleansing:", len(vocab))
        return vocab

    def split_folders(self, files):
        """
        Splits data of dataset into their own lists
        The structure is as follows:
        Dataset
        |-training
        ||-id.ast.sliced  #tree with something missing
        ||-id.ast.slice   #tree of the missing part
        ||-id.methodParameter
        ||-methods
        ||-variables
        ||-activatedSwingElements
        |-validation
        ||-id.ast.sliced
        ||-id.ast.slice
        ||-id.methodParameter
        ||-methods
        ||-variables
        ||-activatedSwingElements
        |-testing
        ||-id.ast.sliced
        ||-id.ast.slice
        ||-id.methodParameter
        ||-methods
        ||-variables
        ||-activatedSwingElements
        """
        training = [path for path in files if os.path.basename(os.path.dirname(path)) == "training"]
        validation = [path for path in files if os.path.basename(os.path.dirname(path)) == "validation"]
        testing = [path for path in files if os.path.basename(os.path.dirname(path)) == "testing"]
        return training, validation, testing

    def split_files(self, dataset):
        """
        splits the given dataset into:
        ast|methodParameter|methods|variables|activatedSwingElements
        """
        # sort the dataset
        dataset = sorted(dataset)
        # get all ast files
        ast_files = [f for f in dataset if f.endswith(".ast.sliced")]
        if self.shuffle:
            random.shuffle(ast_files)
        # a sample contains of all files with the same index
        samples = {}
        samples["ast.sliced"] = []
        samples["ast.slice"] = []
        samples["methodParameter"] = []
        samples["methods"] = []
        samples["variables"] = []
        samples["activatedSwingElements"] = []

        # iterate through all paths
        for ast in ast_files:
            # get the basename
            basename = os.path.basename(ast)
            basepath = "/".join(ast.split("/")[:-1])
            name = basename.split(".")[0]
            # get all other files connected to the current
            slices = basepath + "/" + name + ".ast.slice"
            methodParameter = basepath + "/" + name + ".methodParameter"
            methods = basepath + "/" + name + ".methods"
            variables = basepath + "/" + name + ".variables"
            activatedSwingElements = basepath + "/" + name + ".activatedSwingElements"
            samples["ast.sliced"].append(ast)
            samples["ast.slice"].append(slices)
            samples["methodParameter"].append(methodParameter)
            samples["methods"].append(methods)
            samples["variables"].append(variables)
            samples["activatedSwingElements"].append(activatedSwingElements)
        return samples

    def map_tokens(self, word_array):
        """
        maps the words to the corresponding indices
        sets OoV words to <UNK> and adds them to OoV array
        """
        index_array = []
        oov_array = []
        for word in word_array:
            if word in self.vocab:
                # word is in vocab, append it
                index_array.append(self.word_to_index[word])
            else:
                # word is not in vocab, append <UNK> and add word to oov list
                index_array.append(self.word_to_index["<UNK>"])
                if word not in oov_array:
                    oov_array.append(word)
        return index_array, oov_array

    def pad_sample(self, index_tokens, sample_size):
        """
        pad the sample to the given sample size
        """
        # initialize the result with padding
        result = [self.word_to_index["<PAD>"]] * sample_size
        # put index tokens to result
        for i in range(len(index_tokens)):
            result[i] = index_tokens[i]
        return result
