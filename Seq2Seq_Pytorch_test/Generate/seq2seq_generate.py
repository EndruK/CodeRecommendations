from Seq2Seq.Model.seq2seq_model import LSTMModel
import tensorflow as tf
import os

class Generate:
    def __init__(self, dataset, embedding_model, enc_hidden, dec_hidden):
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.learning_rate = 0.001

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # load model here
        tf.reset_default_graph()
        self.tf_model = LSTMModel(input_size=100,
                                  output_size=100,
                                  batch_size=1,
                                  model=self.dataset,
                                  embed_size=self.embedding_model.embedding_size,
                                  enc_hidden=self.enc_hidden,
                                  dec_hidden=self.dec_hidden,
                                  lr=self.learning_rate)
        self.tf_model.build_graph()

    def restore_checkpoint(self, path):

        print(path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, path)
        # load embeddings
        embedding_feed = {
            self.tf_model.embedding_placeholder: self.embedding_model.final_embeddings}
        self.session.run(self.tf_model.embedding_init, feed_dict=embedding_feed)

    def gen(self):
        assert self.session
        testinputpath = "/media/andre/E896A5A496A573AA/Corpora/AndreKarge_2018-07-02_varDecl_with_Swing_seq2seq/testing/32263.ast.sliced"
        tokens = self.dataset.tokenizer.tokenize(testinputpath, check_embed=False)
        ast_index_tokens, oov_words_ast = self.dataset.map_tokens(tokens)
        ast_index_tokens = self.dataset.pad_sample(ast_index_tokens, self.input_sample_size)
        ast_index_tokens = [ast_index_tokens]


        # push data through model
        feed_dict = {self.tf_model.x: ast_index_tokens}
        [result] = self.session.run([self.softmax_out], feed_dict=feed_dict)
        print(result)

    def close_session(self):
        self.session.close()