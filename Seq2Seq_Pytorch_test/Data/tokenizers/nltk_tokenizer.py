from Seq2Seq_Pytorch_test.Data.tokenizers.tokenizer import Tokenizer
import nltk


class NLTKTokenizer(Tokenizer):

    def tokenize(self, text):
        """
        just a wrapper for the NLTK tokenizer
        :param text: text of a document
        :return: array of tokens of the given text, tokenized via NLTK
        """
        return nltk.word_tokenize(text)
