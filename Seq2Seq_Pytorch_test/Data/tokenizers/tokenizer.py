import abc


class Tokenizer(abc.ABC):

    @abc.abstractmethod
    def tokenize(self, text):
        """
        interface for a tokenizer
        this function should always be called to tokenize a text based on specific schemes
        :param text: text which should be tokenized
        :return: array holding the tokens
        """
        pass
