from Seq2Seq_Pytorch_test.Data.tokenizers.tokenizer import Tokenizer
import json


class JsonTokenizer(Tokenizer):

    def tokenize(self, text):
        """
        tokenize a given json string on json rules (there are lists, dicts and strings in json)
        :param text: json string
        :return: tokenized json string
        """
        json_object = json.loads(text)
        tokens = JsonTokenizer.tokenize_json_string(json_object, [])
        return tokens

    @staticmethod
    def tokenize_json_string(node, token_list):
        """
        traverse tree recursively and extract all tokens based on json rule set
        :param node: current root node
        :param token_list: token array
        :return: token array
        """
        # we can have a list as the current root element - iterate over all list elements
        if isinstance(node, list):
            token_list.append("[")
            for i in range(len(node)):
                item = node[i]
                token_list = JsonTokenizer.tokenize_json_string(item, token_list)
                if i < len(node) - 1:
                    token_list.append(",")
            token_list.append("]")
        # if we have a string - put it in the list
        elif isinstance(node, str):
            # TODO: what to do with backslashes in strings?
            if '\\' in node:
                result = node.replace("\\", "\\\\")
            else:
                result = node
            token_list.append("\"" + result + "\"")
        # if we have a dict, again iterate over items
        elif isinstance(node, dict):
            token_list.append("{")
            cnt = 0
            for key, value in node.items():
                token_list.append("\"" + key + "\"")
                token_list.append(":")
                token_list = JsonTokenizer.tokenize_json_string(value, token_list)
                if cnt < len(node) - 1:
                    token_list.append(",")
                cnt += 1
            token_list.append("}")
        else:
            error_msg = "tree case not implemented: %s" % str(type(node))
            raise NotImplementedError(error_msg)
        return token_list
