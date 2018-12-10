from Seq2Seq_Pytorch_test.Model.seq2seq_pytorch import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Encoder(PytorchEncoder):
    """
    Encoder has the same behaviour as the standard seq2seq encoder
    So, we just call the super class constructor
    """
    def __init__(self, encoder_time_size, encoder_hidden_size, vocab_len, embedding_dim, use_cuda):
        super(Encoder, self).__init__(encoder_time_size, encoder_hidden_size, vocab_len, embedding_dim, use_cuda)


class Decoder(PytorchDecoder):
    """
    Decoder differs from standard seq2seq decoder
    It uses additional Attention weights
    """
    def __init__(self, decoder_hidden_size, vocab_len, embedding_dim, use_cuda):
        super(Decoder, self).__init__(decoder_hidden_size, vocab_len, embedding_dim, use_cuda)
        #self.attention = BahdanauAttention(decoder_hidden_size, use_cuda=use_cuda)
        # self.attention = MyBahdanauAttention(decoder_hidden_size, use_cuda=use_cuda)
        self.attention = BahdanauAttention(decoder_hidden_size, use_cuda=use_cuda)
        self.lstm = nn.LSTM(
            input_size=embedding_dim+decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            bidirectional=False
        )


    def forward(self, x, hidden, encoder_output=None):
        """
        Overrides the Pytorch Decoder forward pass - process one token at a time
        :param x: either the previous ground truth on teacher force or the last prediction
        :param hidden: on first iteration: last encoder state, otherwise last decoder state
        :param encoder_output: all encoder outputs
        :return: output of RNN and last hidden state
        """
        # NOTE: processing one time step at a time
        # we need encoder output here, so assert that it is not none
        assert encoder_output is not None

        # first, embed the input into dense representation
        # x.shape = [batch, 1]
        # embedded.shape = [batch, embed_dim]
        embedded = self.embedding(x)
        # emedded.shape = [time, batch, embed_dim] = [1, batch, embed_dim]
        embedded = embedded.permute(1, 0, 2)

        att_context = self.attention(encoder_output, hidden)
        att_context = att_context.transpose(1,0)
        att_context = torch.cat((embedded.squeeze(0), att_context), -1).unsqueeze(0)
        output, hidden = self.lstm(att_context, hidden)



        # dimensions: Sequence x batch x h => batch x h
        output = output.squeeze(0)
        # project to decoder classes (vocab)
        # finally, project the hidden state to the output classes and softmax
        output = F.log_softmax(self.projection(output), dim=-1)
        return output, hidden

class MyBahdanauAttention(nn.Module):
    def __init__(self, hidden_size, use_cuda=False):
        super(MyBahdanauAttention, self).__init__()
        self.use_cuda = use_cuda
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)


    def forward(self, encoder_output, last_decoder_state):
        """
        :param encoder_outputs: all encoder_outputs, shape [input_length, batch, enc_hidden*2]
        :param last_decoder_state: last decoder state, shape [2, 1, batch, enc_hidden*2] - 1st dim 2 = (h,c)
        :return: attention weights
        """

        # decoder state with time
        state = last_decoder_state[0]

        # eij = v(tanh(W1(state_(i-1))+W2(enc_out_(j))))
        att_score = self.v(torch.tanh(self.w1(encoder_output) + self.w2(state)))
        # aij = softmax(eij)
        att_weights = torch.softmax(att_score, dim=0)
        context_vector = att_weights * encoder_output
        context_vector = torch.sum(context_vector, dim=0)

        return context_vector, att_weights


    def calculate_attention_score(self, encoder_element, decoder_state):
        """

        :param encoder_element:
        :param decoder_state:
        :return:
        """
        # hidden state = (h,c) --> we are interested in the hidden state
        state_h = decoder_state[0].squeeze(0)  # = [3,256] [batch, hidden_size]
        # Attention general:
        attention_score = self.attention(encoder_element)
        # use einsum on batch, since we want element wise dot product for all batches
        attention_score = torch.einsum("ij,ij->i", (state_h, attention_score))
        return attention_score

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, use_cuda=False):
        super(BahdanauAttention, self).__init__()
        self.use_cuda = use_cuda
        self.attention = nn.Linear(hidden_size, hidden_size)


    def forward(self, encoder_output, last_decoder_state):
        """
        :param encoder_outputs: all encoder_outputs, shape [input_length, batch, enc_hidden*2]
        :param last_decoder_state: last decoder state, shape [2, 1, batch, enc_hidden*2] - 1st dim 2 = (h,c)
        :return: attention weights
        """

        # # decoder state with time
        state = last_decoder_state[0]
        batch_size = encoder_output.shape[1]
        #
        # # eij = v(tanh(W1(state_(i-1))+W2(enc_out_(j))))
        # att_score = self.v(torch.tanh(self.w1(encoder_output) + self.w2(state)))
        # # aij = softmax(eij)
        # att_weights = torch.softmax(att_score, dim=0)
        # context_vector = att_weights * encoder_output
        # context_vector = torch.sum(context_vector, dim=0)
        print()
        seq_len = encoder_output.shape[0]
        attention_energies = Variable(torch.zeros(seq_len, batch_size))
        if self.use_cuda: attention_energies = attention_energies.cuda()
        for i in range(seq_len):
            attention_energies[i] = self.calculate_attention_score(state, encoder_output[i])
        sm = torch.softmax(attention_energies, dim=0).unsqueeze(0).unsqueeze(0)
        return sm
        #return context_vector, att_weights


    def calculate_attention_score(self, encoder_element, decoder_state):
        """

        :param encoder_element:
        :param decoder_state:
        :return:
        """
        # # hidden state = (h,c) --> we are interested in the hidden state
        # state_h = decoder_state[0].squeeze(0)  # = [3,256] [batch, hidden_size]
        # # Attention general:
        # attention_score = self.attention(encoder_element)
        # # use einsum on batch, since we want element wise dot product for all batches
        # attention_score = torch.einsum("ij,ij->i", (state_h, attention_score))
        # return attention_score
        energy = self.attention(encoder_element)
        energy = energy.squeeze(0)
        energy = torch.bmm(decoder_state, energy)
        # energy = self.attention(encoder_element)
        # energy = torch.einsum("ij,ij->i", (decoder_state, energy.squeeze(0)))
        return energy


class AttTrainingHelper(TrainingHelper):
    def __init__(self,
                 encoder_time_size,
                 decoder_time_size,
                 hidden_size,
                 vocab_len,
                 embedding_dim,
                 batch_size,
                 learning_rate=0.001,
                 sos=0,
                 eos=1,
                 use_cuda=False,
                 teacher_force_strength=.5,
                 gradient_clipping=5):
        super(AttTrainingHelper, self).__init__(
            encoder_time_size,
            decoder_time_size,
            hidden_size,
            vocab_len,
            embedding_dim,
            batch_size,
            learning_rate,
            sos,
            eos,
            use_cuda,
            teacher_force_strength,
            gradient_clipping)
        self.encoder = Encoder(
            encoder_time_size=self.encoder_time_size,
            encoder_hidden_size=self.encoder_hidden_size,
            vocab_len=self.vocab_len,
            embedding_dim=self.embedding_dim,
            use_cuda=self.use_cuda
        )
        self.decoder = Decoder(
            decoder_hidden_size=self.decoder_hidden_size,
            vocab_len=self.vocab_len,
            embedding_dim=self.embedding_dim,
            use_cuda=self.use_cuda
        )
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def model_iteration(self, x, y, teacher_force=False):
        # build variable tensors out of tuple
        x = Variable(torch.LongTensor(x))
        y = Variable(torch.LongTensor(y))
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()
        # zero gradients of optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # initialize the hidden state of encoder with zero state
        hidden = self.encoder.init_hidden_state(self.batch_size)
        # forward pass encoder
        enc_out, enc_hidden = self.encoder(x, hidden)
        # first decoder state is last encoder state
        # enc_hidden.shape = [2,2,batch,hidden]
        # have to permute it to [1, 2, batch, hidden*2]
        # (h, c)
        dec_hidden_h = torch.cat((enc_hidden[0][0], enc_hidden[1][0]), dim=-1)
        dec_hidden_c = torch.cat((enc_hidden[0][1], enc_hidden[1][1]), dim=-1)

        dec_hidden = torch.stack((dec_hidden_h, dec_hidden_c))
        # we have to add a dimension ...
        dec_hidden = dec_hidden.unsqueeze(1)

        # loss is accumulated for each word
        loss = 0
        # prediction accuracy vector - each element is either 1(correct pred.) or 0(false pred.)
        # since it could be that this has to be a multi-dim array
        acc = Variable(torch.LongTensor([[0.0] * self.batch_size] * self.decoder_time_size))
        if self.use_cuda:
            acc = acc.cuda()

        # the first input of the decoder is the Start of Sequence token
        # shape: [batch, 1]
        decoder_input = self.first_decoder_input

        # NOTE: this part differs from parent method
        attentions = torch.zeros(self.decoder_time_size, self.batch_size, self.encoder_time_size)

        if teacher_force:
            # use ground truth
            for i in range(self.decoder_time_size):
                # forward pass decoder
                dec_out, dec_hidden, attention = self.decoder(decoder_input, dec_hidden, enc_out)
                # NOTE: this part differs from parent method
                attention = attention.squeeze(-1).transpose(1, 0)
                attentions[i] = attention


                # acc[i] = accuracy for the current timestep
                _, top_index = dec_out.data.topk(1)
                acc[i] = top_index.squeeze(-1) == y[:, i]
                # get the next y as input and add a dimension
                decoder_input = y[:, i].unsqueeze(-1)
                label = y[:, i]
                # calc loss and accumulate
                loss += self.loss_function(dec_out, label)
        else:
            # use decoder prediction
            for i in range(self.decoder_time_size):
                # forward pass decoder
                dec_out, dec_hidden, attention = self.decoder(decoder_input, dec_hidden, enc_out)
                # NOTE: this part differs from parent method
                attention = attention.squeeze(-1).transpose(1, 0)
                attentions[i] = attention


                # get top value of prediction
                top_value, top_index = dec_out.data.topk(1)
                acc[i] = top_index.squeeze(-1) == y[:, i]
                # build tensor out of index
                if self.use_cuda:
                    decoder_input = Variable(torch.cuda.LongTensor(top_index))
                else:
                    decoder_input = Variable(torch.LongTensor(top_index))
                label = y[:, i]
                # calc loss
                loss += self.loss_function(dec_out, label)
                if self.use_cuda:
                    decoder_input = decoder_input.cuda()

        # TODO: accuracy only on true sequence (not padding)

        batch_loss = loss.item() / self.decoder_time_size
        if self.use_cuda:
            batch_accuracy = np.mean(acc.cpu().numpy())
        else:
            batch_accuracy = np.mean(acc.numpy())

        # return mean loss for all words in prediction
        return loss, batch_loss, batch_accuracy, attentions

    def training_iteration(self, x, y):
        # calc teacher force prob
        teacher_force = random() < self.teacher_force_strength

        # run model iteration
        loss, batch_loss, batch_accuracy, attention = self.model_iteration(x, y, teacher_force)

        # backpropagation
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        # application of new weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # return mean loss for all words in prediction
        return batch_loss, batch_accuracy, attention

    def valid_test_iteration(self, x, y):
        # deactivate gradient calculation
        with torch.no_grad():
            # run model iteration (we don't need the loss here since no backpropagation)
            # also we don't want teacher force
            _, batch_loss, batch_accuracy, attention = self.model_iteration(x, y, teacher_force=False)
            # return mean loss for all words in prediction
        return batch_loss, batch_accuracy, attention

    def plot_attention(input_sentence, output_sentence, attention_weights):
        """
        plots an attention weight matrix
        :param input_sentence: the input sequence as a list string
        :param output_sentence: the output sequence as a list of strings
        :param attention_weights: [input_size, output_size]
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_ax = ax.matshow(attention_weights, cmap="bone")
        fig.colorbar(show_ax)

        # TODO: remove this
        input_sentence = [str(x) for x in input_sentence]
        output_sentence = [str(x) for x in output_sentence]

        ax.set_yticklabels([''] + input_sentence, rotation=90)
        ax.set_xticklabels([''] + output_sentence)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.show()
        plt.close()


# # TODO: testing code - remove this later
# if __name__ == "__main__":
#     input_time = 5
#     output_time = 7
#     batch_size = 3
#     hidden_size = 128
#     vocab_len = 50
#     embedding_dim = 512
#
#     x = [[1] * input_time] * batch_size
#     y = [[2] * output_time] * batch_size
#     x = Variable(torch.LongTensor(x)).cuda()
#     y = Variable(torch.LongTensor(y)).cuda()
#
#     enc = Encoder(input_time, hidden_size, vocab_len, embedding_dim, True).cuda()
#     dec = Decoder(hidden_size*2, vocab_len, embedding_dim, True).cuda()
#
#     hidden = enc.init_hidden_state(batch_size)
#     # context = Variable(torch.zeros((input_time, batch_size), hidden_size*2)).cuda()
#
#     # run
#     for i in range(50):
#         attentions = torch.zeros(output_time, batch_size, input_time)
#         enc_out, enc_hidden = enc(x, hidden)
#         # first decoder state is last encoder state
#         # enc_hidden.shape = [2,2,batch,hidden]
#         # have to permute it to [1, 2, batch, hidden*2]
#         # (h, c)
#         dec_hidden_h = torch.cat((enc_hidden[0][0], enc_hidden[1][0]), dim=-1)
#         dec_hidden_c = torch.cat((enc_hidden[0][1], enc_hidden[1][1]), dim=-1)
#
#         dec_hidden = torch.stack((dec_hidden_h, dec_hidden_c))
#         # we have to add a dimension ...
#         dec_hidden = dec_hidden.unsqueeze(1)
#         dec_input = Variable(torch.LongTensor([[12]]*batch_size)).cuda()
#         dec_result = torch.zeros(output_time, batch_size)
#         for j in range(output_time):
#             dec_out, dec_hidden, attn = dec(dec_input, dec_hidden, enc_out)
#             batch_indices = dec_out.argmax(dim=-1)
#             dec_result[j] = batch_indices
#             dec_input = y[:,j].unsqueeze(-1)
#             attn = attn.squeeze(-1).transpose(1,0)
#             attentions[j] = attn
#         dec_result = dec_result.transpose(1, 0)
#         attentions = attentions.transpose(1,0).transpose(2,1) #[batch, in_size, out_size]
#         plot_attention(x[0].cpu().data.numpy(), dec_result[0].cpu().data.numpy(), attentions[0].cpu().data.numpy())
#         print()