import torch, numpy as np, os, logging as log
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from random import random
from shutil import copyfile
import torch.nn.functional as F


#######################################################
# Seq2Seq Pytorch splitted into two modules enc and dec
# and Helper which models a training iteration
#######################################################

class PytorchEncoder(nn.Module):
    """
    Standard Seq2Seq Encoder as a pytorch module
    """
    def __init__(self, encoder_time_size, encoder_hidden_size, vocab_len, embedding_dim, use_cuda):
        super(PytorchEncoder, self).__init__()

        # LSTM variables
        self.encoder_time_size = encoder_time_size
        self.encoder_hidden_size = encoder_hidden_size

        # embedding variables
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            self.vocab_len,
            self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.encoder_hidden_size,
            bidirectional=True
        )
        self.use_cuda = use_cuda

    def forward(self, x, hidden):
        """
        forward pass - process the complete input sequence at a time
        :param x: input sequence
        :param hidden: initialized hidden state (zero state)
        :return: result and hidden state of decoder
        """
        # NOTE: processing full input sequence in one run
        # x.shape = [batch, time, 1] - 1 dim is index of vocab
        embedded = self.embedding(x)
        # embedded.shape = [time, batch, embed_dim]
        embedded = embedded.permute(1,0,2)
        output, last_hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden_state(self, batch_size):
        """
        initialize the hidden state with a zero tensor (no knowledge present)
        :param batch_size:
        :return: zero hidden state
        """
        # [2, 2, batch, hidden]
        # 1.dim=2 because of bi-directional
        # 2.dim=2 because LSTM =(h, c)
        hidden = Variable(torch.zeros(2, 2, batch_size, self.encoder_hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden


class PytorchDecoder(nn.Module):
    """
    default Seq2Seq Decoder as a pytorch module without attention
    """
    def __init__(self, decoder_hidden_size, vocab_len, embedding_dim, use_cuda):
        super(PytorchDecoder, self).__init__()

        self.decoder_hidden_size = decoder_hidden_size

        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_len,
            embedding_dim=self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.decoder_hidden_size,
            bidirectional=False
        )
        self.projection = nn.Linear(
            in_features=self.decoder_hidden_size,
            out_features=self.vocab_len
        )
        self.use_cuda = use_cuda

    def forward(self, x, hidden, encoder_output=None):
        """
        forward pass - process one token at a time
        :param x: either the previous ground truth on teacher force or the last prediction
        :param hidden: on first iteration: last encoder state, otherwise last decoder state
        :param encoder_output: relevant parameter for attention
        :return: output of RNN and last hidden state
        """
        # NOTE: processing one time step at a time

        # first, embed the input into dense representation
        # x.shape = [batch, 1]
        # embedded.shape = [batch, embed_dim]
        embedded = self.embedding(x)
        # emedded.shape = [time, batch, embed_dim] = [1, batch, embed_dim]
        embedded = embedded.permute(1, 0, 2)
        # pass embedded input through LSTM unit
        output, hidden = self.lstm(embedded, hidden)
        # dimensions: Sequence x batch x h => batch x h
        output = output.squeeze(0)
        # project to decoder classes (vocab)
        # finally, project the hidden state to the output classes and softmax
        output = F.log_softmax(self.projection(output), dim=-1)
        return output, hidden


class TrainingHelper:
    """
    Class to help the model for training - models a training iteration for the pytorch model
    """
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
        """
        constructor of the training helper to define parameters for all training steps
        :param encoder_time_size: how long is the input sequence
        :param decoder_time_size: how long is the output sequence
        :param hidden_size: size of the hidden layer (= double for decoder)
        :param vocab_len: how many classes does the vocab have
        :param embedding_dim: dense dimension of the embedding space
        :param batch_size:
        :param learning_rate:
        :param sos: index of sos token in the vocabulary
        :param eos: index of eos token in the vocabulary
        :param teacher_force_strength: probability of forcing teacher decoder (default=0.5)
        :param gradient_clipping: prevent gradient explosion (default=5)
        """
        self.encoder_time_size = encoder_time_size
        self.decoder_time_size = decoder_time_size
        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size * 2
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.teacher_force_strength = teacher_force_strength
        self.clip = gradient_clipping
        self.learning_rate = learning_rate
        self.use_cuda = use_cuda

        # start of sequence token (default = index 0)
        self.sos = sos
        self.first_decoder_input = Variable(torch.LongTensor([[self.sos]] * self.batch_size))
        if self.use_cuda:
            self.first_decoder_input = self.first_decoder_input.cuda()

        # end of sequence token (default = index 1)
        self.eos = eos
        self.eos_batch_check = Variable(torch.LongTensor([[self.eos]] * self.batch_size))
        if self.use_cuda:
            self.eos_batch_check = self.eos_batch_check.cuda()

        self.encoder = PytorchEncoder(
            encoder_time_size=self.encoder_time_size,
            encoder_hidden_size=self.encoder_hidden_size,
            vocab_len=self.vocab_len,
            embedding_dim=self.embedding_dim,
            use_cuda=self.use_cuda
        )
        self.decoder = PytorchDecoder(
            decoder_hidden_size=self.decoder_hidden_size,
            vocab_len=self.vocab_len,
            embedding_dim=self.embedding_dim,
            use_cuda=self.use_cuda
        )
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        # negative log likelihood loss - loss to train classification with C classes
        # https://pytorch.org/docs/stable/nn.html#nllloss
        self.loss_function = nn.NLLLoss()  # TODO: parametrize
        #self.loss_function = nn.CrossEntropyLoss()  # TODO: parametrize

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)  # TODO: parametrize
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)  # TODO: parametrize

    def model_iteration(self, x, y, teacher_force=False):
        """
        one model iteration (complete pass through)
        without backpropagation, to reuse this code for validation and testing
        :param x:
        :param y:
        :param teacher_force:
        :return:
        """
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
        if teacher_force:
            # use ground truth
            for i in range(self.decoder_time_size):
                # forward pass decoder
                dec_out, dec_hidden = self.decoder(decoder_input, dec_hidden)
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
                dec_out, dec_hidden = self.decoder(decoder_input, dec_hidden)
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
        return loss, batch_loss, batch_accuracy

    def training_iteration(self, x, y):
        """
        one iteration for training
        uses a probability to either force teacher decoder input (ground truth) or pass generated logits to
        next time-step
        :param x: input sequence batch
        :param y: label sequence batch
        :return: mean loss of output sequence and mean accuracy of output sequence
        """

        # calc teacher force prob
        teacher_force = random() < self.teacher_force_strength

        # run model iteration
        loss, batch_loss, batch_accuracy = self.model_iteration(x, y, teacher_force)

        # backpropagation
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)
        # application of new weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # return mean loss for all words in prediction
        return batch_loss, batch_accuracy

    def valid_test_iteration(self, x, y):
        """
        one iteration for validation and testing (without backpropagation)
        :param x: input sequence batch
        :param y: output sequence batch
        :return: mean loss of output sequence and mean accuracy of output sequence
        """
        # deactivate gradient calculation
        with torch.no_grad():
            # run model iteration (we don't need the loss here since no backpropagation)
            # also we don't want teacher force
            _, batch_loss, batch_accuracy = self.model_iteration(x, y, teacher_force=False)
            # return mean loss for all words in prediction
        return batch_loss, batch_accuracy

    def store_model(self, path, name, store_model=False):
        """
        store the model weights to the disk, copy the model file and write
        the best checkpoint name to a file
        :param path: path to folder where model should be stored
        :param name: name of the dump file
        :param store_model: flag to also copy the model file to checkpoint path
        :return:
        """
        # save the weights
        if not os.path.isdir(path):
            log.debug("checkpoint directory not found - creating new folder at " + path)
            os.makedirs(path)
        torch.save(self.encoder, os.path.join(path, "encoder." + name))
        torch.save(self.decoder, os.path.join(path, "decoder." + name))
        if store_model and not os.path.isfile(os.path.join(path, "model.py")):
            # copy the model file
            log.debug("copy model file to checkpoint path: " + path + "/best_checkpoint")
            copyfile(os.path.abspath(__file__), os.path.join(path, "model.py"))
        with open(os.path.join(path, "best_checkpoint"), "w") as f:
            f.write(name)


    def load_model(self, checkpoint_path, checkpoint_name):
        """
        load the weights from a file
        :param checkpoint_path: path to checkpoint
        :param checkpoint_name: name of checkpoint
        :return:
        """
        log.info("loading checkpoint from disk")
        if not os.path.isfile(os.path.join(checkpoint_path, "encoder." + checkpoint_name)) \
                and not os.path.isfile(os.path.join(checkpoint_path, "decoder." + checkpoint_name)):
            log.error("weights not found! - aborting")
            log.debug("checkpoint_path = " + checkpoint_path + " model_name = " + checkpoint_name)
            return
        self.encoder = torch.load(os.path.join(checkpoint_path, "encoder." + checkpoint_name))
        self.decoder = torch.load(os.path.join(checkpoint_name, "decoder." + checkpoint_name))
        log.info("loaded checkpoint")
