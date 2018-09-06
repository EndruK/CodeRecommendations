import re
from nltk.tokenize.regexp import RegexpTokenizer


class RegexTokenizer:
    def __init__(self):
        pattern = r'''(?x)
            (?:(\-\s\w+\(\w+\=\w+\)\:))  # match any: - word(word):
            | (?:(w[0-9]+))                # match any: w42
            | (?:(<INV>))                  # match invocation token
            | (?:(\w+\(\w+\=\w+\)\:))      # match any: word(word=word)
            | (?:(\w+\:\s\"\w+\"))         # match any: word:"word"
            | (?:(\w+\:\s\"[.,\/#!$%\^&\*;:{}=\-_`~()]+\")) # match any: word:"punctuation"
            | (?:(\w+\:\s\"\w+\s\w+\"))    # match any: word:"word word word"
            | (?:(\w+\:))                  # match any: word:
            | (?:(\w+\:\s\".*\"))          # match the rest with magic strings in there
        '''
        self.regex = RegexpTokenizer(pattern)

    def tokenize(self, input, check_embed=False):
        """
        interface for this tokenizer
        only takes full paths as input
        if the incoming file is an AST -> use the ast tokenizer
        """
        check_strings = [".ast.sliced", "ast.slice"]
        if check_embed:
            check_strings = [".ast"]

        with open(input) as f:
            lines = f.readlines()

        check_passed = False
        for check in check_strings:
            if input.endswith(check):
                check_passed = True
                break

        if check_passed:
            return self.tokenize_ast(lines)
        else:
            return self.tokenize_feature(lines)

    def tokenize_ast(self, lines):
        """
        regex tokenization
        can handle full paths or text as input
        """
        all_tokens = []
        # iterate over the given lines
        for line in lines:
            # tokenize line
            line_tokens = self.regex.tokenize(line)
            all_tokens.extend(line_tokens)
        result = []
        # check for line breaks and length
        for token in all_tokens:
            for subtoken in token:
                if len(subtoken) > 0:
                    result.append(subtoken.strip())
        return result

    def tokenize_feature(self, lines):
        """
        normal feature tokenization
        """
        result = []
        for line in lines:
            # strip the line from unwanted stuff
            stripped = line.strip()
            # a feature consists of at least 3 characters
            if len(stripped) > 2:
                pair = stripped.split(",")
                feature_type, feature_identifier = pair
                result.append((feature_type, feature_identifier))
        return result