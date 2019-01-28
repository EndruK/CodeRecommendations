from Seq2Seq_Pytorch_test.Data.tokenizers.tokenizer import Tokenizer
import nltk


class NLTKTokenizer(Tokenizer):

    def tokenize(self, text):
        """
        just a wrapper for the NLTK tokenizer but with different behaviour for quotes
        `` -> ", '' -> "
        :param text: text of a document
        :return: array of tokens of the given text, tokenized via NLTK
        """
        tokens = nltk.word_tokenize(text)
        result = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "``":
                result.append('"')
            elif tokens[i] == "''":
                if tokens[i-1] != "\"" and tokens[i+1] != "\"":
                    result.append('"')
            else:
                result.append(tokens[i])
            i += 1
        return result
