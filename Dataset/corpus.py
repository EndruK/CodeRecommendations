import os, random

class Corpus:
    def __init__(self, path_to_corpus, partition_scheme, shuffle_files=False):
        self.path = path_to_corpus
        self.partition_scheme = partition_scheme

        self.corpus_file_list = self.create_file_list(shuffle_files)

    def create_file_list(self, shuffle=False):
        """
        get the absolute path of all AST files in the corpus
        :param shuffle: shuffle the files?
        :return: list of all paths
        """
        paths = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".ast"):
                    paths.append(os.path.join(root, file))
        if shuffle:
            random.shuffle(paths)
        return paths
