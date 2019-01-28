import os, random, logging as log, json, multiprocessing, pickle as pkl, csv, pandas as pd, math
# TODO: remove this import
from tokenizers.nltk_tokenizer import NLTKTokenizer as Tokenizer
from Seq2Seq_Pytorch_test.Data.partition import Partition


class Dataset:

    TARGET_AST_NODE = "statements"
    INVOCATION_STR = "{<INV>:<EMPTY>}"
    STATEMENT_LIMIT = 100


    def __init__(self, Tokenizer):
        """

        :param Tokenizer: tokenizer class reference
        """
        self.pandas_dataset = None
        self.partitions = {}
        self.tokenizer = Tokenizer()




    def preprocess(self, corpus_path, cut_result_path, preprocess_result_path, num_of_processes=5):
        """
        Main preprocessing procedure - the function will take some time!
        TODO: reorder this
        - extract all files in corpus
        - split the dataset into partitions
        - create vocabulary
        - build index mappings
        - create the AST cut
        :param corpus_path: string with absolute path to the corpus root folder
        :return: TODO
        """
        log.debug("start preprocessing")
        # extract file paths
        file_paths = self.get_corpus_file_list(corpus_path, shuffle=True)

        # create AST cuts
        Dataset.extract_statements_parallel(file_list=file_paths,
                                            output_path=cut_result_path,
                                            num_of_processes=num_of_processes)
        Dataset.build_csv_file(path_to_cut=cut_result_path, output_path=preprocess_result_path)

    def load_csv(self, csv_path):
        """
        load the dataset csv via pandas dataframe
        :param csv_path:
        :return:
        """
        log.info("start loading csv file")
        self.pandas_dataset = pd.read_csv(csv_path)
        log.info("done loading csv file")


    def partition_dataset(self, preprocess_result_path, scheme=[.7, .2, .1]):
        """
        separate the dataset into training, validation, testing
        :param preprocess_result_path:
        :return:
        """
        log.info("start building partitions")
        complete_length = len(self.pandas_dataset)

        size_training = math.floor(scheme[0] * complete_length)
        size_validation = math.floor(scheme[1] * complete_length)

        index_list = [i for i in range(complete_length)]
        random.shuffle(index_list)

        training_indices = index_list[:size_training]
        validation_indices = index_list[size_training:size_training+size_validation]
        testing_indices = index_list[size_training+size_validation:]

        training_partition = Partition("training", self.pandas_dataset, training_indices)
        validation_partition = Partition("validation", self.pandas_dataset, validation_indices)
        testing_partition = Partition("testing", self.pandas_dataset, testing_indices)

        self.partitions = {"training" : training_partition,
                           "validation" : validation_partition,
                           "testing" : testing_partition}
        log.info("done building partitions")


    # @staticmethod
    # def build_csv_file(path_to_cut, output_path):
    #     """
    #     put everything into one large csv file
    #     :param path_to_cut: path to the x,y tuple files
    #     :param output_path: path to where to store the csv file
    #     :return:
    #     """
    #
    #     file_list = os.listdir(path_to_cut)
    #     file_list = [os.path.join(path_to_cut, f) for f in file_list if f.endswith(".pkl")]
    #
    #     csv_file_path = os.path.join(output_path, "dataset.csv")
    #     if not os.path.isdir(output_path):
    #         os.makedirs(output_path)
    #
    #     csv_file = open(csv_file_path, "w")
    #     writer = csv.writer(csv_file, lineterminator="\n")
    #
    #     writer.writerow(["x", "y"])
    #
    #     index = 0
    #     file_cnt = 1
    #     for file in file_list:
    #         with open(file, "rb") as f:
    #             data = pkl.load(f)
    #         for x,y in data:
    #             if index % 200 == 0:
    #                 print("file %d of %d - processing tuple %d" % (file_cnt, len(file_list), index))
    #             t = [x, y]
    #             writer.writerow(t)
    #             index += 1
    #         file_cnt += 1
    #     csv_file.close()


    # # TODO: obolete, remove this
    # @staticmethod
    # def split_dataset(complete_cut_dataset_path, output_path, scheme=[.7, .2, .1]):
    #     """
    #     split the dataset into training, validation, testing by using the given scheme
    #     :param scheme: splitting scheme to divide the dataset (default=[.7, .2, .1])
    #     """
    #
    #     if not os.path.isdir(output_path):
    #         os.makedirs(output_path)
    #
    #     # load the summarization file
    #     with open(os.path.join(complete_cut_dataset_path, "corpus.details") , "r") as f:
    #         text = f.read()
    #     sum_of_tuples = int(text.split(":")[1])
    #
    #     training_sum = int(sum_of_tuples * scheme[0])
    #     validation_sum = int(sum_of_tuples * scheme[1])
    #     testing_sum = int(sum_of_tuples * scheme[2])
    #
    #     # now do the split
    #     file_list = os.listdir(complete_cut_dataset_path)
    #     file_list = [os.path.join(complete_cut_dataset_path, f) for f in file_list if f.endswith(".pkl")]
    #
    #     counter = 0
    #
    #     training_file = open(os.path.join(output_path, "training.partition.pkl"), "wb")
    #     validation_file = open(os.path.join(output_path, "validation.partition.pkl"), "wb")
    #     testing_file = open(os.path.join(output_path, "testing.partition.pkl"), "wb")
    #
    #     for file in file_list:
    #         with open(file, "rb") as f:
    #             content = pkl.load(f)
    #         random.shuffle(content)
    #         for tuple in content:
    #             if counter % 200 == 0 and counter > 0:
    #                 print("%d / %d" % (counter, sum_of_tuples))
    #             if counter < training_sum:
    #                 pkl.dump(tuple, training_file)
    #             elif counter >= training_sum and counter < training_sum + validation_sum:
    #                 pkl.dump(tuple, validation_file)
    #             else:
    #                 pkl.dump(tuple, testing_file)
    #             counter += 1
    #     training_file.close()
    #     validation_file.close()
    #     testing_file.close()
    #     with open(os.path.join(output_path, "details.txt"), "w") as f:
    #         f.write("training:%d\nvalidation:%d\ntesting:%d\n" % (training_sum, validation_sum, testing_sum))



    # @staticmethod
    # def extract_statements_parallel(file_list, output_path, num_of_processes=5):
    #     """
    #     create processes for statement extraction
    #     takes very long time on single HDD
    #     :param file_list: array containing the absolute paths to all files in the corpus
    #     :param output_path: path to store the result of the statement cuts
    #     :param num_of_processes: number of parallel processes
    #     """
    #     # how many files should one process handle
    #     file_load = len(file_list) // num_of_processes
    #     # how many files are left
    #     file_load_rest = len(file_list) % num_of_processes
    #     file_split = [file_load] * num_of_processes
    #     for i in range(file_load_rest):
    #         file_split[i] += 1
    #     assert sum(file_split) == len(file_list)
    #
    #     # create a process pool
    #     pool = multiprocessing.Pool(processes=num_of_processes)
    #     # array to store process hooks
    #     processes = []
    #     log.debug("start statement extraction processes")
    #     for process_id in range(num_of_processes):
    #         start = sum(file_split[:process_id])
    #         end   = start + file_split[process_id]
    #         # get the sublist of files
    #         process_file_list = file_list[start:end]
    #         # spawn a process
    #         process = pool.apply_async(Dataset.extract_statement_process,
    #                                    args=(process_file_list, process_id, output_path))
    #         processes.append(process)
    #     process_tuple_count = []
    #     # wait for all processes to finish
    #     for process in processes:
    #         tuple_count = process.get()
    #         process_tuple_count.append(tuple_count)
    #     log.info("final amount of tuples of this run = %d" % sum(process_tuple_count))
    #     with open(os.path.join(output_path, "corpus.details"), "w") as f:
    #         f.write("amount_of_tuples:%d" % sum(process_tuple_count))
    #     log.debug("statement extraction processes finished")


    # @staticmethod
    # def extract_statement_process(file_list, process_id, output_path, num_per_file=20000):
    #     """
    #     process to work with a sublist of files for statement extraction
    #     :param file_list: sublist of AST files
    #     :param process_id: the id of the process running this method
    #     :param output_path: path where the result should be exported to
    #     :param num_per_file: max number of tuples per dump file
    #     :return: tuple_count of all extracted tuples of this process
    #     """
    #     log.debug("started process %d" % process_id)
    #     source_file_count = 1
    #     tuple_count = 0
    #
    #     tuple_array = []
    #
    #     start_range = 1
    #
    #     for path in file_list:
    #         if source_file_count % 200 == 0:
    #             log.debug("process %d running file %d/%d" % (process_id, source_file_count, len(file_list)))
    #         # get statement tuples
    #         with open(path, "r") as f:
    #             json_object = json.load(f)
    #         tuples = Dataset.extract_statements_file(json_object, limit=Dataset.STATEMENT_LIMIT)
    #         for i in range(len(tuples)):
    #             t = tuples[i]
    #             tuple_array.append(t)
    #             tuple_count += 1
    #
    #
    #             if tuple_count % num_per_file == 0:
    #                 end_range = tuple_count
    #                 name = "%d_%s_%s.pkl" % (process_id, start_range, end_range)
    #                 log.debug("creating new statement cut file %s at %s for process %d" % (name,
    #                                                                                        output_path,
    #                                                                                        process_id))
    #                 file_path = os.path.join(output_path, name)
    #                 with open(file_path, "wb") as f:
    #                     pkl.dump(tuple_array, f)
    #                 tuple_array = []
    #                 start_range = tuple_count+1
    #
    #         source_file_count += 1
    #     if len(tuple_array) > 0:
    #         end_range = tuple_count
    #         name = "%d_%s_%s.pkl" % (process_id, start_range, end_range)
    #         log.debug("creating new statement cut file %s at %s for process %d" % (name,
    #                                                                                output_path,
    #                                                                                process_id))
    #         file_path = os.path.join(output_path, name)
    #         with open(file_path, "wb") as f:
    #             pkl.dump(tuple_array, f)
    #     return tuple_count

    @staticmethod
    def extract_statements_file(method_json_object, limit=100):
        """
        Extract statements of a given file.
        Read, tokenize and then slice some statements out of the text of a given file
        :param source_path: path to the file to extract statement
        :param limit: limit the amount of statements per file (default=100)
        :return: array holding x and y tuples
        """
        # get all statements of the text
        statement_lists = Dataset.process_json_node(method_json_object, target=Dataset.TARGET_AST_NODE)
        result_statements = []
        for statement_list in statement_lists:
            for statement in statement_list:
                result_statements.append(statement)
        # shuffle our statement list
        random.shuffle(result_statements)
        result = []
        count = 0

        json_string = str(method_json_object)

        # build x and y pairs
        for statement in result_statements:
            if count >= limit:
                break
            found = json_string.find(str(statement))
            if found == -1:
                continue

            x = json_string
            y = str(statement)
            x = x.replace(y, Dataset.INVOCATION_STR)

            result.append((x, y))
            count += 1

        return result

    # @staticmethod
    # def extract_statements_file_old(source_path, limit=100):
    #     """
    #     TODO: obsolete - remove this
    #     Extract statements of a given file.
    #     Read, tokenize and then slice some statements out of the text of a given file
    #     :param source_path: path to the file to extract statement
    #     :param limit: limit the amount of statements per file (default=100)
    #     :return: array holding x and y tuples
    #     """
    #     # first read in the file
    #     text = Dataset.read_file(source_path)
    #     # get all statements of the text
    #     statements = Dataset.slice_method_statements(text)
    #     if len(statements) > limit:
    #         end = limit
    #     else:
    #         end = len(statements)
    #     # shuffle our statement list
    #     random.shuffle(statements)
    #     result = []
    #     # build x and y pairs
    #     for statement in statements[:end]:
    #         json_str = str(json.loads(text))
    #         stmt_str = str(statement)
    #         # replace the part in the original source with the invocation string
    #         # TODO: at this point something goes wrong and the strings are not replaced properly
    #         x = json_str.replace(stmt_str, Dataset.INVOCATION_STR)
    #         y = stmt_str
    #         # x and y as tuple
    #         result.append((x, y))
    #     return result

    # @staticmethod
    # def slice_method_statements(text):
    #     """
    #     Build json object and get all expression statements by using recursive function process_node.
    #     :param text: text of the given AST file
    #     :return: list containing all statments of that file
    #     """
    #     json_data = json.loads(text)
    #     statements = Dataset.process_json_node(json_data, target=Dataset.TARGET_AST_NODE)
    #     return statements

    @staticmethod
    def process_json_node(node, target, target_list=[]):
        """
        Recursive function to extract a certain target node type of a json tree
        :param node: root node of the json tree
        :param target: the target json node which should be extracted
        :param target_list: container for the results
        :return target_list: at the end of recursion this should hold all subtrees with target as root
        :raises NotImplementedError: only if no case was covered by the recursive function
        """
        # we can have a list as the current root element - iterate over all list elements
        if isinstance(node, list):
            for item in node:
                Dataset.process_json_node(item, target, target_list)
        # if we have a string - ignore it
        elif isinstance(node, str):
            pass
        # if we have a dict, again iterate ofer items
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == target:
                    # print("### " + str(key) + "  " + str(value))
                    target_list.append(value)
                target_list = Dataset.process_json_node(value, target, target_list)
        else:
            error_msg = "tree case not implemented: %s" % str(type(node))
            raise NotImplementedError(error_msg)
        return target_list

    # @staticmethod
    # def process_json_node(node, target, target_list=[]):
    #     """
    #     Recursive function to extract a certain target node type of a json tree
    #     :param node: root node of the json tree
    #     :param target: the target json node which should be extracted
    #     :param target_list: container for the results
    #     :return target_list: at the end of recursion this should hold all subtrees with target as root
    #     :raises NotImplementedError: only if no case was covered by the recursive function
    #     """
    #     # we can have a list as the current root element - iterate over all list elements
    #     if isinstance(node, list):
    #         for item in node:
    #             Dataset.process_json_node(item, target, target_list)
    #     # if we have a string - ignore it
    #     elif isinstance(node, str):
    #         pass
    #     # if we have a dict, again iterate ofer items
    #     elif isinstance(node, dict):
    #         for key, value in node.items():
    #             if key == target:
    #                 target_list.append(node)
    #             target_list = Dataset.process_json_node(value, target, target_list)
    #     else:
    #         error_msg = "tree case not implemented: %s" % str(type(node))
    #         log.error(error_msg)
    #         raise NotImplementedError(error_msg)
    #     return target_list





    def get_corpus_file_list(self, corpus_path, shuffle=False):
        """
        Extract all necessary file of the dataset.
        Each entry is an absolute path to one .ast file
        :param corpus_path: string with absolute path to corpus root folder
        :param shuffle: set true to shuffle paths in array
        :return: array containing all paths
        """
        result = []
        for root, dirs, files in os.walk(corpus_path):
            for file in files:
                if file.endswith(".ast"):
                    path = os.path.join(root, file)
                    result.append(path)
        log.debug("done extracting corpus paths")
        log.info("number of files in corpus: %d" % len(result))
        # only if we want to shuffle our dataset
        if shuffle:
            random.shuffle(result)
        return result

    # def create_vocab_parallel(self, file_paths, num_of_threads):
    #     """
    #     Create the vocabulary based on the class tokenizer and the given files.
    #     Process the files in parallel to speed up the generation.
    #     :param file_paths: array containing the absolute paths to all files in the corpus
    #     :param num_of_threads: number of threads to process the corpus
    #     :return:
    #     """

    # def create_vocab(self, file_paths):
    #     """
    #     Create the vocabulary based on the class tokenizer and the given files.
    #     Function may take some time.
    #     :param file_paths: array containing the absolute paths to all files in the corpus
    #     :return: dictionary containing all words with their occurences in the corpus {"word": occurence}
    #     """
    #     vocab = {}
    #
    #     log.debug("start creation of vocabulary")
    #
    #     for path in file_paths:
    #         text = Dataset.read_file(path)
    #         tokens = self.tokenizer.tokenize(text)
    #         for token in tokens:
    #             if token not in vocab:
    #                 vocab[token] = 1
    #             else:
    #                 vocab[token] += 1
    #     log.info("done extracting all words in corpus, num of words: %d" % len(vocab))
    #     print()


    @staticmethod
    def read_file(path):
        """
        simply read a text file and return its content
        :param path: absolute path to the file
        :return: string containing the complete content of the given file
        :raises FileNotFoundException: only if the path does not exists or the file cannot be read
        """
        if not os.path.isfile(path):
            log.error("%s does not exist!")
            raise FileNotFoundError

        with open(path, "r") as f:
            text = f.read()
        return text


# TODO: test code
if __name__ == "__main__":
    log.basicConfig(
        level=log.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            log.StreamHandler()
        ]
    )

    ds = Dataset(Tokenizer)
    corpus_path =     "/mnt/media/Corpora/AndreKarge_2018-09-12_JavaAST_JSON_without_features/Dataset"
    cut_result_path = "/mnt/media/Corpora/AndreKarge_2018-09-12_JavaAST_JSON_without_features/2019_01_09_split_ast_methods_statement_v3"
    #preprocess_result_path = "/mnt/media/Corpora/AndreKarge_2018-09-12_JavaAST_JSON_without_features/csv_test"
    preprocess_result_path = "/home/andre/Documents"
    num_of_processes = 7
    # ds.preprocess(corpus_path=corpus_path,
    #               cut_result_path=cut_result_path,
    #               preprocess_result_path=preprocess_result_path,
    #               num_of_processes=num_of_processes)
    #
    ds.load_csv(csv_path=preprocess_result_path+"/test.cvs")
    ds.partition_dataset(preprocess_result_path=preprocess_result_path)
    test = ds.partitions["training"][0]

    pass