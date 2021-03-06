import torch.utils.data as data
import torch.nn.utils.rnn as rnn
from typing import List
from typing import Dict
import torch
import numpy as np


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
        Sentences are still words (JSON format).

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
        x = self.tokenize_sentence(x) + [self.w2i["EOS"]]
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
            x = self.tokenize_sentence(x) + [self.w2i["EOS"]]
            y = self.tokenize_sentence(y) + [self.w2i["EOS"]]
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

    def collate3(self, batch):
        """
        Call this function using the pytorch DataLoader
        https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

        :param batch: result of __getitem__ inside of an array with the size of the batch
        :return: what each iteration of dataloader should return
                 in this case: padded x index sequences
        """
        # first, tokenize sentences and translate them to index sequences
        indices = []
        for x, y in batch:
            # append EOS token
            x_tokens = self.tokenize_sentence(x) + ["EOS"]
            y_tokens = self.tokenize_sentence(y) + ["EOS"]
            x_indices = torch.LongTensor(self.word_sequence_to_index_sequence(x_tokens))
            y_indices = torch.LongTensor(self.word_sequence_to_index_sequence(y_tokens))
            indices.append([x_indices, y_indices])
        # sort sequences by x sequence length in descending order
        sorted_batch = sorted(indices, key=lambda k: len(k[0]), reverse=True)
        x = [b[0] for b in sorted_batch]
        y = [b[1] for b in sorted_batch]
        # get the longest y sequence length
        max_target_len = max([len(el) for el in y])
        # pad input and output to longest sequence
        x_padded = rnn.pad_sequence(x, batch_first=True, padding_value=self.w2i["PAD"])
        y_padded = rnn.pad_sequence(y, batch_first=True, padding_value=self.w2i["PAD"])
        # put the input and output lengths into tensors
        x_lengths = torch.Tensor([len(element) for element in x])
        y_lengths = [len(element) for element in y]
        # create a masking matrix for output sequence
        y_mask = np.zeros(shape=y_padded.shape)
        for i in range(len(batch)):
            y_mask[i, :y_lengths[i]] = 1.0
        # transpose all tensors (from [batch, time] to [time, batch])
        y_mask = torch.ByteTensor(y_mask.transpose(1, 0))
        x_padded = x_padded.transpose(1, 0)
        y_padded = y_padded.transpose(1, 0)
        return x_padded, x_lengths, y_padded, y_mask, max_target_len

