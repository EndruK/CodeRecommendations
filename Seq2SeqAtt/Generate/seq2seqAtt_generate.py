from Seq2SeqAtt.Model.seq2seqAtt_model import AttentionModel
import tensorflow as tf
import os

class Generate:
    def __init__(self, dataset, embedding_model, enc_hidden, dec_hidden):
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # load model here
        tf.reset_default_graph()
        self.batch_size = 32 # 32
        self.model = AttentionModel(
            input_size=100,
            output_size=100,
            batch_size=self.batch_size,
            model=dataset,
            embed_size=self.embedding_model.embedding_size,
            enc_hidden_size=self.enc_hidden,
            dec_hidden_size=self.dec_hidden,
            enc_layer_depth=1,
            dec_layer_depth=1,
            lr=0.001
        )
        self.model.build_graph()
        print(self.dataset.word_to_index["<PAD>"])
        print(self.dataset.index_to_word[5319])

    def restore_checkpoint(self, path):
        print(path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, path)
        #embedding_feed = {self.model.embedding_placeholder: self.embedding_model.final_embeddings}
        #self.session.run(self.model.embedding_init, feed_dict=embedding_feed)

    def gen(self):
        assert self.session
        testinputpath = "/media/andre/E896A5A496A573AA/Corpora/AndreKarge_2018-07-02_varDecl_with_Swing_seq2seq/testing/32263.ast.sliced"
        tokens = self.dataset.tokenizer.tokenize(testinputpath, check_embed=False)
        ast_index_tokens, oov_words_ast = self.dataset.map_tokens(tokens)
        ast_index_tokens = self.dataset.pad_sample(ast_index_tokens, 100)
        ast_index_tokens = [ast_index_tokens] * self.batch_size

        # push data through model
        feed_dict = {self.model.x: ast_index_tokens,
                     self.model.is_training: False,
                     self.model.y: [[0]*100]*self.batch_size}
        [result] = self.session.run([self.model.pred_argmax], feed_dict=feed_dict)
        result = [self.dataset.index_to_word[i] for i in result[0]]
        print(result)

    def close_session(self):
        self.session.close()