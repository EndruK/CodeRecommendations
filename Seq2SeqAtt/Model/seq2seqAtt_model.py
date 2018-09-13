import tensorflow as tf


class AttentionModel:
    def __init__(self,
                 input_size,
                 output_size,
                 batch_size,
                 model,
                 embed_size,
                 enc_hidden_size,
                 enc_layer_depth,
                 dec_hidden_size,
                 dec_layer_depth,
                 lr):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.model = model
        self.embed_size = embed_size
        self.enc_hidden_size = enc_hidden_size
        self.enc_layer_depth = enc_layer_depth
        self.dec_hidden_size = dec_hidden_size
        self.dec_layer_depth = dec_layer_depth
        self.attention_size = 60
        self.lr = lr

    def build_graph(self):
        self.build_graph_variables()
        with tf.name_scope("encoder"):
            with tf.variable_scope("encoder"):
                encoder_out, encoder_state = self.encoder(self.x)
        # with tf.name_scope("attention"):
        #     with tf.variable_scope("attention"):
        #         attention_context_vector = self.attention(encoder_out, encoder_state)
        with tf.name_scope("decoder"):
            with tf.variable_scope("decoder"):
                self.decoder_out, self.attention_weights = self.decoder(
                    encoder_out, encoder_state, self.y)
        with tf.name_scope("metric"):
            self.metric(self.decoder_out, self.y, self.y_masks)

    def build_graph_variables(self):
        # Network Input
        self.x = tf.placeholder(tf.int32, [None, self.input_size], name="x_input")

        # Embeddings
        self.W = tf.Variable(
            tf.constant(0.0, shape=[len(self.model.vocab), self.embed_size]),
            trainable=False, name="W")
        self.embedding_placeholder = tf.placeholder(
            tf.float32, [len(self.model.vocab), self.embed_size])
        self.embedding_init = self.W.assign(self.embedding_placeholder)

        # Attention variables
        self.W_attention1 = tf.layers.Dense(self.dec_hidden_size)
        self.W_attention2 = tf.layers.Dense(self.dec_hidden_size)
        self.V_attention = tf.layers.Dense(1)

        # Decoder variables
        self.projection = tf.layers.Dense(len(self.model.vocab))
        self.first_dec_input = tf.Variable([self.model.word_to_index["<GO>"]] * self.batch_size, dtype=tf.int32,
                                           name="first_dec_input")

        # Inference
        self.y = tf.placeholder(tf.int32, [None, self.output_size], name="labels")
        self.y_masks = tf.placeholder(tf.float32, [None, self.output_size], name="label_masks")

        # training switch
        self.is_training = tf.placeholder(tf.bool, [], name="training_inference_switch")

    def encoder(self, x):
        """
        Build the seq2seq encoder.
        Input:
            X: Tensor[batch_size, time]
        Output:
            encoder_final_out: Tensor[batch_size, time, encoder_hidden_size*2]
            encoder_final_state:
                LSTMStateTuple<c=[batch_size, time, encoder_hidden_size*2],h=[batch_size, time, encoder_hidden_size*2]>
        """
        # first, embed the x input
        x = tf.nn.embedding_lookup(self.W, x)

        encoder_fw_cell = tf.contrib.rnn.BasicLSTMCell(
            self.enc_hidden_size, dtype=tf.float32)
        encoder_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            self.enc_hidden_size, dtype=tf.float32)
        # init hidden states with zero
        encoder_fw_state_init = encoder_fw_cell.zero_state(
            self.batch_size, dtype=tf.float32)
        encoder_bw_state_init = encoder_bw_cell.zero_state(
            self.batch_size, dtype=tf.float32)

        # push it to a bidirectional dynamic RNN
        # time major = False leads to output shape: [batch_size, time, encoder_hidden_size]
        # swap memory = True leads to Transparently swap the tensors produced in
        # forward inference but needed for back prop from GPU to CPU.
        # This allows training RNNs which would typically not fit on a
        # single GPU, with very minimal (or no) performance penalty
        (encoder_out_fw, encoder_out_bw), (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            encoder_fw_cell,
            encoder_bw_cell,
            x,
            initial_state_fw=encoder_fw_state_init,
            initial_state_bw=encoder_bw_state_init,
            time_major=False,
            swap_memory=True
        )
        # concat fw and bw final output
        encoder_final_out = tf.concat(
            [encoder_out_fw, encoder_out_bw], axis=2, name="encoder_output")
        encoder_state_h = tf.concat(
            [encoder_state_fw.h, encoder_state_bw.h], axis=1, name="encoder_state_h")
        encoder_state_c = tf.concat(
            [encoder_state_fw.c, encoder_state_bw.c], axis=1, name="encoder_state_c")
        encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
            c=encoder_state_c, h=encoder_state_h)

        return encoder_final_out, encoder_final_state

    def attention(self, encoder_output, decoder_state):
        """
        build the attention weight
        Based on Bahdanau Attention
        Input:
            encoder_final_out: Tensor[batch_size, time, encoder_hidden_size*2]
            encoder_final_state:
                LSTMStateTuple<c=[batch_size, time, encoder_hidden_size*2],h=[batch_size, time, encoder_hidden_size*2]>
        Output:
            attention_context_vector: Tensor[batch_size, encoder_hidden_size*2]
        """
        # encoder state h is the last hidden state
        # shape: [batch_size, encoder_hidden_size*2]
        # but we need [batch_size, 1, encoder_hidden_size*2]
        decoder_state_with_time = tf.expand_dims(decoder_state.h, axis=1)

        # Bahdanau additive style to compute the attention score: tanh(W1(ht)+W2(hs))
        # shape: [batch_size, time, encoderhidden_size*2]
        attention_score = tf.nn.tanh(
            self.W_attention1(encoder_output) + self.W_attention2(decoder_state_with_time))

        # score is passed through a 1 unit dense layer and the weights are retreived via softmax
        # shape: [batch_size, time, 1]
        attention_weights = tf.nn.softmax(self.V_attention(attention_score), axis=1)

        # attention_weights = tf.Print(attention_weights, [attention_weights])

        # now we multiply the weights with the encoder output to get our context vector
        # shape: [batch_size, time, encoder_hidden_size*2]
        attention_context_vector = attention_weights * encoder_output

        # summarize the context
        # shape: [batch_size, encoder_hidden_size*2]
        attention_context_vector = tf.reduce_sum(attention_context_vector, axis=1)

        return attention_context_vector, attention_weights

    def decoder(self, encoder_output, encoder_state, y_input):
        """
        own decoder
        based on #https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/
        Input:
            encoder_final_out: Tensor[batch_size, time, encoder_hidden_size*2]
            encoder_final_state:
                LSTMStateTuple<c=[batch_size, time, encoder_hidden_size*2],h=[batch_size, time, encoder_hidden_size*2]>
            attention_context_vector: Tensor[batch_size, encoder_hidden_size*2]
        Output:
            decoder_projected: Tensor[batch_size, time, vocab_length]
        """
        decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.dec_hidden_size, dtype=tf.float32)

        output_ta = tf.TensorArray(size=self.output_size, dtype=tf.int32)
        # input_ta = tf.TensorArray(size=self.output_size, dtype=tf.int32)
        # input_ta = input_ta.unstack(tf.transpose(y_input, [1, 0]))
        # print(input_ta)
        att = tf.TensorArray(size=self.output_size, dtype=tf.float32)

        # define the decoder loop
        def decoder_loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output
            if cell_output is None:
                # first loop call - init variables
                next_cell_state = encoder_state
                # embed the input
                next_input = tf.nn.embedding_lookup(
                    self.W, self.first_dec_input)
                attention_context, attention_weights = self.attention(
                    encoder_output, next_cell_state)
                # shape: [time, batch, embed]
                next_input = tf.concat([attention_context, next_input], axis=1)
                next_loop_state = att
                next_loop_state = next_loop_state.write(
                    time, attention_weights)
            else:
                # next state = current state
                next_cell_state = cell_state
                # next input
                # first project that cell output
                # then embed that projected output
                next_input = tf.cond(self.is_training,
                                     # lambda: input_ta.read(time-1),
                                     lambda: y_input[:, time - 1],
                                     lambda: tf.cast(tf.argmax(self.projection(cell_output), axis=1), dtype=tf.int32))
                next_input = tf.nn.embedding_lookup(self.W, next_input)
                attention_context, attention_weights = self.attention(
                    encoder_output, next_cell_state)
                # finally concat attention with embedded projected output
                next_input = tf.concat(
                    [attention_context, next_input], axis=1)
                next_loop_state = tf.cond(time >= self.output_size,
                                          lambda: loop_state,
                                          lambda: loop_state.write(time, attention_weights))
                # next_loop_state = loop_state.write(time, attention_weights)
            finished = (time >= self.output_size)
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)

        decoder_emit_ta, _, att_out = tf.nn.raw_rnn(decoder_cell, decoder_loop_fn)
        # stack the tensor array and reshape [time, batch, vocab_len] -> [batch, time, vocab_len]
        decoder_out = tf.transpose(decoder_emit_ta.stack(), [1, 0, 2])

        attention_weights = att_out.stack()
        attention_weights = tf.squeeze(
            tf.transpose(attention_weights, [1, 0, 2, 3]), -1)

        # project the output
        decoder_projected = self.projection(decoder_out)
        # print(decoder_projected)
        return decoder_projected, attention_weights

    def metric(self, dec_out, y, y_masks):
        """
        evaluate the model
        """
        # print(dec_out)
        # print(y)
        crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=dec_out)
        # print(crossentropy)
        self.loss = tf.reduce_mean(crossentropy)
        # self.loss = (tf.reduce_sum(crossentropy*tf.stack(y_masks)) / (self.batch_size*self.output_size))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.pred_argmax = tf.cast(tf.argmax(dec_out, axis=2), tf.int32)
        self.batch_accuracy = tf.reduce_sum(tf.cast(tf.equal(self.pred_argmax, y), tf.float32) * y_masks) / tf.cast(
            tf.reduce_sum(y_masks), tf.float32)
