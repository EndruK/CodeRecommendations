import nltk

class NLTKTokenizer:
    def __init__(self):
        pass

    def tokenize(self, input, is_path=False):
        """
        basic nltk word tokenizer
        can handle full paths or text as input
        """
        if is_path:
            with open(input) as f:
                text = f.read()
        else:
            text = input
        tokens = nltk.word_tokenize(text)
        tokens = self.join_inv_token(tokens)
        return tokens

    def join_inv_token(self, tokens):
        """
        join the tokens "<", "INV", ">" together
        """
        # word tokenizer slices <INV>
        for i, t in enumerate(tokens):
            if t == "INV" and i > 0 and i < len(tokens) - 1:
                if tokens[i - 1] == "<" and tokens[i + 1] == ">":
                    new_token = tokens[i - 1] + tokens[i] + tokens[i + 1]
                    tokens = tokens[:i - 1] + [new_token] + tokens[i + 2:]
                    tokens[i - 1] = new_token
                    break
        return tokens