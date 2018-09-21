import tensorflow as tf
from tensorflow.python.layers.layers import Dense

class LSTMModel:
    def __init__(self,
                 input_size,
                 output_size,
                 batch_size,
                 model,
                 embed_size,
                 enc_hidden,
                 dec_hidden,
                 lr):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.model = model
        self.embed_size = embed_size
        self.enc_hidden = enc_hidden
        self.dec_hidden = 2 * enc_hidden
        self.lr = lr

    def build_graph(self):
        # https://github.com/udacity/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb

        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.int32, [self.batch_size, self.input_size])
            self.y = tf.placeholder(tf.int32, [self.batch_size, self.output_size])
            self.y_masks = tf.placeholder(tf.float32, [self.batch_size, self.output_size])
            self.W = tf.Variable(tf.constant(0.0, shape=[len(self.model.vocab), self.embed_size]), trainable=False, name='W')
            self.embedding_placeholder = tf.placeholder(tf.float32, [len(self.model.vocab), self.embed_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.embedding_lookup_x = tf.nn.embedding_lookup(self.W, self.x)
            self.embedding_lookup_y = tf.nn.embedding_lookup(self.W, self.y)
            self.is_training = tf.placeholder(tf.bool, [], name="training_inference_switch")
        with tf.name_scope("encoder"):
            encoder_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.enc_hidden, dtype=tf.float32)
            encoder_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.enc_hidden, dtype=tf.float32)
            encoder_fw_state = encoder_fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            encoder_bw_state = encoder_bw_cell.zero_state(self.batch_size, dtype=tf.float32)

            (encoder_out_fw, encoder_out_bw), (encoder_out_state_fw, encoder_out_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                encoder_fw_cell,
                encoder_bw_cell,
                self.embedding_lookup_x,
                initial_state_fw=encoder_fw_state,
                initial_state_bw=encoder_bw_state,
                time_major=False,
                swap_memory=True
            )
            #encoder_out = tf.concat([encoder_out_fw, encoder_out_bw], axis=2, name="encoder_out")
            encoder_state = tf.contrib.rnn.LSTMStateTuple(
                c=tf.concat([encoder_out_state_fw.c, encoder_out_state_bw.c], axis=1, name="encoder_state_c"),
                h=tf.concat([encoder_out_state_fw.h, encoder_out_state_bw.h], axis=1, name="encoder_state_h")
            )
        with tf.name_scope("decoder"):
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_hidden, dtype=tf.float32)
            projection = Dense(len(self.model.vocab), use_bias=True, name="output_projection")
            with tf.variable_scope("decoder"):
                training_helper = tf.contrib.seq2seq.TrainingHelper(self.embedding_lookup_y,
                                                                    sequence_length=[self.output_size for _ in
                                                                                     range(self.batch_size)])
                training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                   training_helper,
                                                                   encoder_state,
                                                                   projection)
                self.training_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                     impute_finished=True,
                                                                                     maximum_iterations=self.output_size)
            with tf.variable_scope("decoder", reuse=True):
                start_tokens = tf.tile(tf.constant([self.model.w2i["<GO>"]], dtype=tf.int32),
                                       [self.batch_size],
                                       name="start_tokens")
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.W,
                                                                            start_tokens,
                                                                            self.model.w2i["<PAD>"])
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                    inference_helper,
                                                                    encoder_state,
                                                                    projection)
                self.inference_decoder_output,_,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                                      impute_finished=True,
                                                                                      maximum_iterations=self.output_size)

        with tf.name_scope("metric"):
            logits = self.training_decoder_output.rnn_output
            crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=logits)
            # add masks to loss to ignore padding
            self.loss = (tf.reduce_sum(crossentropy * tf.stack(self.y_masks)) / (self.batch_size * self.output_size))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            with tf.name_scope("validation"):
                self.pred_argmax = tf.cast(tf.argmax(logits, 2), dtype=tf.int32)
                # apply y_masks to prevent padding influence
                # self.batch_accuracy = tf.reduce_mean(
                #     tf.cast(tf.equal(pred_argmax, self.y), tf.float32)*self.y_masks)
                self.batch_accuracy = tf.reduce_sum(tf.cast(tf.equal(self.pred_argmax, self.y), tf.float32) *
                                                    self.y_masks) / tf.cast(tf.reduce_sum(self.y_masks), tf.float32)
        with tf.name_scope("inference_metric"):
            i_logits = self.inference_decoder_output.rnn_output
            self.inference_argmax = tf.cast(tf.argmax(i_logits, 2), dtype=tf.int32)
            self.inference_batch_accuracy = tf.reduce_sum(tf.cast(tf.equal(self.inference_argmax, self.y), tf.float32) *
                                                          self.y_masks) / tf.cast(tf.reduce_sum(self.y_masks), tf.float32)


