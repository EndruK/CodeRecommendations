import torch.utils.data as data
from typing import List
from typing import Dict


class Partition(data.Dataset):
    def __init__(self, x, y, tokenizer):
        """
        Initializes the dataset with given symmetric arrays.

        :param x: x-array created by sklearn train_test split
        :param y: y-array created by sklearn train_test split
        :param tokenizer: Class reference to the used tokenizer
        """
        super(Partition, self).__init__()
        assert(len(x) == len(y))
        self.length = len(x)
        self.x = x
        self.y = y
        self.vocab = None  # type: List[str]
        self.w2i = None  # type: Dict[str, int]
        self.tokenizer = tokenizer()

    def __len__(self):
        """
        Get the length of the partition.

        :return: int length
        """
        return self.length

    def __getitem__(self, item):
        """
        Get an item out of the partition.

        :param item: index to the desired tuple
        :return: tuple of x and y (tokenized sentences)
        """
        return self.x[item], self.y[item]

    def set_vocab_and_mapping(self, vocab, w2i):
        """
        Sets the vocabulary and index mapping of a partition.

        :param vocab: the complete vocab for index translation
        :param w2i: word to index mapping
        """
        self.vocab = vocab
        self.w2i = w2i

    def word_sequence_to_index_sequence(self, word_sequence):
        """
        Translate a list of word tokens into a list of indices.

        :param word_sequence: the word sequence list
        :return: the index sequence list
        """
        assert self.vocab is not None
        assert self.w2i is not None
        result = []
        for word_token in word_sequence:
            if word_token in self.vocab:
                result.append(self.w2i[word_token])
            else:
                result.append(self.w2i["UNK"])
        result.append(self.w2i["EOS"])
        return result

    def tokenize_sentence(self, sentence):
        """
        Simply tokenizes a given sentence based on the initialized tokenizer

        :param sentence: input sequence
        :return: tokenized sequence
        """
        return self.tokenizer.tokenize(sentence)

    def collate_single(self, x):
        """
        Collate function for a single x

        :param x: input sequence
        :return: index array in a 1 batch
        """
        x = self.tokenize_sentence(x)
        x = self.word_sequence_to_index_sequence(x)
        return x

    def collate(self, batch):
        """
        Function to use as collate for torch.utils.data.DataLoader.
        Tokenize the given sequences in the batch.
        Pads x and y elements to the length of the longest element in the batch.
        Creates a masking for y to encode padded tokens.

        :param batch: a word batch containing [x, y] * batch_size (result of __getitem__)
        :return: an index batch containing [x, y, padding] * batch_size - with padding
        """
        l_x = 0
        i_x = -1
        l_y = 0
        i_y = -1
        # tokenize the batches and
        # get the longest length of the current batch for x and y
        for i in range(len(batch)):
            x, y = batch[i]
            x = self.tokenize_sentence(x)
            y = self.tokenize_sentence(y)
            if len(x) > l_x:
                l_x = len(x)
                i_x = i
            if len(y) > l_y:
                l_y = len(y)
                i_y = i
            batch[i] = x, y
        resulting_batch = []
        # PAD all elements shorter than the longest up to the longest size
        # also create a masking for y
        for i in range(len(batch)):
            # create an element with [x, y, mask] for the resulting batch
            x, y = batch[i]
            if i != i_x:
                x_size = len(x)
                x = x + ["PAD"] * (l_x - x_size)
            if i != i_y:
                y_size = len(y)
                y = y + ["PAD"] * (l_y - y_size)
                mask = [1.0] * y_size + [0.0] * (l_y - y_size)
            else:
                mask = [1.0] * l_y
            resulting_batch.append([self.word_sequence_to_index_sequence(x),
                                    self.word_sequence_to_index_sequence(y),
                                    mask])
        return resulting_batch
