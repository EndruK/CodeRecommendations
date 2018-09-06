class Seq2SeqSampleData:
    """
    Samples the given dataset to create batches which can be used
    as input for the neural network.

    TODO:
    - how large is a sample? - Input parameter
    - what to do with data, that has more words than a sample size?

    """

    def __init__(self,
                 dataset,
                 input_sample_size,
                 output_sample_size,
                 batch_size):
        self.dataset = dataset
        self.input_sample_size = input_sample_size
        self.output_sample_size = output_sample_size
        self.batch_size = batch_size

    def get_batch(self, dataset):
        """
        gets the next batch for the dataset
        """
        sample_generator = self.get_sample(dataset)
        while True:
            x_batch = []
            f_batch = []
            y_batch = []
            y_mask_batch = []
            i = 0
            while i < self.batch_size:
                try:
                    x, f, y, y_mask, oov_in, oov_out = next(sample_generator)
                    x_batch.append(x)
                    f_batch.append(f)
                    y_batch.append(y)
                    y_mask_batch.append(y_mask)
                except Exception as e:
                    # sample generator at end - end of batches
                    return
                i += 1
            yield x_batch, f_batch, y_batch, y_mask_batch

    def get_sample(self, dataset):
        """
        generator for a sample
        """
        # get the ast files
        for i in range(len(dataset["ast.sliced"])):
            # read all files corresponding to the index
            ast_file = dataset["ast.sliced"][i]
            label_file = dataset["ast.slice"][i]
            method_param_file = dataset["methodParameter"][i]
            method_file = dataset["methods"][i]
            variable_file = dataset["variables"][i]
            swing_element_file = dataset["activatedSwingElements"][i]

            # tokenize x and y
            ast_word_tokens = self.dataset.tokenizer.tokenize(ast_file, check_embed=False)
            label_word_tokens = self.dataset.tokenizer.tokenize(label_file, check_embed=False)
            # translate x and y to index array
            ast_index_tokens, oov_words_ast = self.dataset.map_tokens(ast_word_tokens)
            label_index_tokens, oov_words_label = self.dataset.map_tokens(label_word_tokens)
            # TODO: change this later for oversized samples
            if (len(ast_index_tokens) > self.input_sample_size
                    or len(label_index_tokens) > self.output_sample_size):
                continue
            ast_index_tokens = self.dataset.pad_sample(ast_index_tokens, self.input_sample_size)
            label_index_tokens = self.dataset.pad_sample(label_index_tokens, self.output_sample_size)
            y_mask = self.mask_y(label_index_tokens, self.output_sample_size)
            # label_index_tokens = self.one_hot_encode(label_index_tokens)
            # TODO: add features
            features = []
            yield ast_index_tokens, features, label_index_tokens, y_mask, oov_words_ast, oov_words_label
            if i == len(dataset["ast.sliced"]) - 1:
                i = 0

    def mask_y(self, labels, size):
        result = [0.0] * size
        pad_token = self.dataset.word_to_index["<PAD>"]
        for i in range(len(labels)):
            if labels[i] != pad_token:
                result[i] = 1.0
            else:
                break
        return result

    def one_hot_encode(self, label_array):
        """
        one hot encode a given array
        """
        result = []
        for l in label_array:
            one_hot = [0] * len(self.dataset.vocab)
            one_hot[l] = 1
            result.append(one_hot)
        return result
