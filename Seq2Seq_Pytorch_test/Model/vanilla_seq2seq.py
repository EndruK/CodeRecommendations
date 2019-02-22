import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import os


class VanillaSeq2Seq:
    def __init__(self,
                 hidden_size,
                 batch_size,
                 vocab_size,
                 embedding_dimension,
                 cuda_enabled,
                 sos_index,
                 eos_index,
                 teacher_force_probability=0.5,
                 gradient_clipping_limit=5,
                 learning_rate=0.001):
        """
        Initialize the Vanilla Sequence 2 Sequence model.
        This model contains of an encoder and a decoder with LSTM units.

        :param hidden_size: the hidden size for encoder and decoder (decoder_hidden = 2 * encoder_hidden)
        :param batch_size: how many items per iteration
        :param vocab_size: amount of target classes in the vocabulary
        :param embedding_dimension: dimension of the embedding space
        :param cuda_enabled: Flag whether to use GPU or CPU
        :param sos_index: index in vocabulary to the start of sequence token
        :param teacher_force_probability: probability to force a ground truth teacher decoding iteration (0.0-1.0)
            (default=0.5)
        :param gradient_clipping_limit: Clip gradients at this value to prevent gradient explosion (default=5)
        :param learning_rate: good old learning rate (default=0.001)
        """
        # model parameter
        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = 2 * self.encoder_hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.cuda_enabled = cuda_enabled
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.teacher_force_probability = teacher_force_probability
        self.gradient_clipping_limit = gradient_clipping_limit
        self.learning_rate = learning_rate

        self.loss_function = nn.NLLLoss()

        self.encoder = Encoder(
            hidden_size=self.encoder_hidden_size,
            vocab_size=self.vocab_size,
            embedding_dimension=self.embedding_dimension,
            cuda_enabled=self.cuda_enabled
        )

        self.decoder = Decoder(
            hidden_size=self.decoder_hidden_size,
            vocab_size=self.vocab_size,
            embedding_dimension=self.embedding_dimension,
            cuda_enabled=self.cuda_enabled
        )
        if self.cuda_enabled:
            self.decoder = self.decoder.cuda()
            self.encoder = self.encoder.cuda()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

    def model_iteration(self, x, y, mask, teacher_force=False):
        """
        The main part for a complete pass through of the model for a iteration.

        :param x: input sequence batch of shape: [batch, time]
        :param y: target sequence batch of shape: [batch, time]
        :param mask: target sequence masking batch of shape: [batch, time]
        :param teacher_force: Flag to force teaching (use of the target tokens as input for decoder)
        :return: tuple containing the mean batch loss and the mean batch accuracy
        """
        x = Variable(torch.LongTensor(x))
        y = Variable(torch.LongTensor(y))
        if self.cuda_enabled:
            x = x.cuda()
            y = y.cuda()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        hidden = self.encoder.init_hidden_state(self.batch_size)
        encoder_output, encoder_last_hidden_state = self.encoder(x, hidden)
        decoder_hidden_h = torch.cat((encoder_last_hidden_state[0][0], encoder_last_hidden_state[1][0]), dim=-1)
        decoder_hidden_h = decoder_hidden_h.unsqueeze(0)
        decoder_hidden_c = torch.cat((encoder_last_hidden_state[0][1], encoder_last_hidden_state[1][1]), dim=-1)
        decoder_hidden_c = decoder_hidden_c.unsqueeze(0)
        decoder_hidden = (decoder_hidden_h, decoder_hidden_c)
        # reshape y from (batch, time) to (time, batch)
        y = y.permute(1, 0)  # shape: (time, batch)
        decoder_input = [self.sos_index] * self.batch_size  # shape: (batch)
        if self.cuda_enabled:
            decoder_input = torch.cuda.LongTensor(decoder_input)
        else:
            decoder_input = torch.LongTensor(decoder_input)


        loss = 0
        acc = Variable(torch.LongTensor([[0.0] * self.batch_size] * len(y)))
        if self.cuda_enabled:
            acc = acc.cuda()
        # if we want to use ground truth for enhanced training - set teacher force=True
        if teacher_force:
            # use ground truth as input
            for i in range(len(y)):
                # use ground truth as input
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                _, top_index = decoder_output.data.topk(1)
                acc[i] = top_index.squeeze(-1) == y[i]
                decoder_input = y[i]
                labels = y[i]  # shape: (batch)
                loss += self.loss_function(decoder_output, labels)
        else:
            # use predictions as input
            for i in range(len(y)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                # decoder_output shape: (batch, vocab)
                top_value, top_index = decoder_output.data.topk(1)  # top_index shape: (batch, 1)
                acc[i] = top_index.squeeze(-1) == y[i]
                top_index = top_index.squeeze(-1)  # shape: (batch)
                if self.cuda_enabled:
                    decoder_input = Variable(torch.cuda.LongTensor(top_index))
                    decoder_input = decoder_input.cuda()
                else:
                    decoder_input = Variable(torch.LongTensor(top_index))
                labels = y[i]
                loss += self.loss_function(decoder_output, labels)
        batch_loss = loss.item() / len(y)

        if self.cuda_enabled:
            #batch_accuracy = np.mean(acc.cpu().numpy())
            acc_numpy = acc.cpu().numpy()
        else:
            #batch_accuracy = np.mean(acc.numpy())
            acc_numpy = acc.numpy()

        # we have to change the dimensions of masking from [batch, time] to [time, batch]
        mask = np.swapaxes(mask, 1, 0)

        # summarize the masking array to get the amount of unmasked elements
        unmasked_count = np.sum(mask, axis=0)
        # apply masking to accuracy array and summarize the correct hits
        hit_count = np.sum((acc_numpy * mask), axis=0)
        # get the masked accuracy for all batch elements
        batch_accuracy = hit_count / unmasked_count
        # get the mean accuracy for the current batch
        mean_batch_accuracy = np.mean(batch_accuracy)

        return loss, batch_loss, mean_batch_accuracy

    def training_iteration(self, batch):
        """
        A Training iteration with backpropagation.

        :param batch: contains all the data of one batch in the shape: [batch, x-y-m, length]
            [:, 0] = x
            [:, 1] = y
            [:, 2] = mask
            len(y) == len(mask) != len(x)
        :return: tuple containing the mean batch loss and the mean batch accuracy
        """
        batch = np.array(batch)
        x = np.array(batch[:, 0].tolist())
        y = np.array(batch[:, 1].tolist())
        m = np.array(batch[:, 2].tolist())
        teacher_force = random.random() < self.teacher_force_probability
        loss, batch_loss, batch_accuracy = self.model_iteration(x, y, m, teacher_force)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clipping_limit)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clipping_limit)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return batch_loss, batch_accuracy

    def validation_iteration(self, batch):
        """
        A validation iteration without backpropagation,

        :param batch: contains all the data of one batch in the shape: [batch, x-y-m, length]
            [:, 0] = x
            [:, 1] = y
            [:, 2] = mask
            len(y) == len(mask) != len(x)
        :return: tuple containing the mean batch loss and the mean batch accuracy
        """
        batch = np.array(batch)
        x = np.array(batch[:, 0].tolist())
        y = np.array(batch[:, 1].tolist())
        m = np.array(batch[:, 2].tolist())
        with torch.no_grad():
            _, batch_loss, batch_accuracy = self.model_iteration(x, y, m, teacher_force=False)
        return batch_loss, batch_accuracy

    def generation_iteration(self, x, limit=1000):
        """
        One iteration for a trained model for und user generation.

        :param x: the input sequence in the shape [time]
        :param limit: how many tokens should be generated until we cancel
        :return: the resulting index array in shape of [time]
        """
        with torch.no_grad():
            # since x should be batchless - add the dimension for the batch here
            x = torch.LongTensor([x])
            if self.cuda_enabled:
                x = x.cuda()
            hidden = self.encoder.init_hidden_state(1)
            encoder_output, encoder_last_hidden_state = self.encoder(x, hidden)
            decoder_hidden_h = torch.cat((encoder_last_hidden_state[0][0], encoder_last_hidden_state[1][0]), dim=-1)
            decoder_hidden_h = decoder_hidden_h.unsqueeze(0)
            decoder_hidden_c = torch.cat((encoder_last_hidden_state[0][1], encoder_last_hidden_state[1][1]), dim=-1)
            decoder_hidden_c = decoder_hidden_c.unsqueeze(0)
            decoder_hidden = (decoder_hidden_h, decoder_hidden_c)
            decoder_input = [self.sos_index]
            if self.cuda_enabled:
                decoder_input = torch.cuda.LongTensor(decoder_input)
            else:
                decoder_input = torch.LongTensor(decoder_input)
            generated_index = -1
            result = []
            # NOTE: this is nearly the same procedure as the no-teacher version in model_iteration
            while True:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                top_value, top_index = decoder_output.data.topk(1)
                top_index = top_index.squeeze(-1)  # shape: (batch)
                if self.cuda_enabled:
                    decoder_input = Variable(torch.cuda.LongTensor(top_index))
                    decoder_input = decoder_input.cuda()
                else:
                    decoder_input = Variable(torch.LongTensor(top_index))
                generated_index = top_index.squeeze(0).data.item()  # shape: (1)
                result.append(generated_index)
                # TODO: change 5 to eof token index
                if generated_index == self.eos_index or len(result) > limit:
                    break
            return result

    def save(self, path, name, acc):
        """
        Stores the model weights to disk at the given path with the given name.
        In addition, the accuracy of this state will be documented in a text file next to the stored weights.

        :param path: destination path (absolute - should be checked for existence)
        :param name: name of the dump
        :param acc: validation accuracy of the dump
        """
        assert os.path.isdir(path)
        torch.save(self.encoder, os.path.join(path, "encoder." + name))
        torch.save(self.decoder, os.path.join(path, "decoder." + name))
        with open(os.path.join(path, "details.md"), "w") as f:
            f.write("validation accuracy of model: %2.4f\n" % acc)

    def load(self, path, name):
        """
        Loads the model weights from disk
        :param path: path of the stored weights (absolute - should be checked for existence)
        :param name: name of the dump
        """
        assert os.path.isdir(path)
        assert os.path.isfile(os.path.join(path, "encoder." + name))
        assert os.path.isfile(os.path.join(path, "decoder." + name))
        self.encoder = torch.load(os.path.join(path, "encoder." + name))
        self.decoder = torch.load(os.path.join(path, "decoder." + name))
        self.encoder.eval()
        self.decoder.eval()


class Encoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, embedding_dimension, cuda_enabled):
        """
        Initialize the Seq2Seq Encoder.
        NOTE: this module takes the complete input sequence and generates an intermediate representation for it.

        :param hidden_size: the dimension of the hidden state
        :param vocab_size: how many classes are in the vocabulary
        :param embedding_dimension: dimension of the embedding space
        :param cuda_enabled: flag to define if calculation should happen on GPU instead of CPU
        """
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dimension
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dimension,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.cuda_enabled = cuda_enabled

    def forward(self, x, hidden):
        """
        Forward pass for one call of this module.
        NOTE: this module processes the input sequence completely in one iteration!

        :param x: input sequence shape: [batch, time]
        :param hidden: the dimension of the hidden state
        :return: tuple(output, hidden) of shape: output[time, batch, hidden_size], hidden[2, 2, batch, hidden_size]
        """
        # x shape: (batch, time)
        # NOTE: processing full input sequence in one run
        x_embedded = self.embedding(x)  # shape: (batch, time, embedding)
        # LSTM expects inputs of shape (time, batch, input_size) - so permute
        x_embedded = x_embedded.permute(1, 0, 2)  # shape: (time, batch, embedding)
        output, last_hidden_state = self.lstm(x_embedded, hidden)
        return output, last_hidden_state

    def init_hidden_state(self, batch_size):
        """
        Initialize a hidden state for the LSTM unit.
        LSTM state = (h, c)
        h = hidden_part = (num_layers * num_directions, batch, hidden_size)
        c = hidden_part = (num_layers * num_directions, batch, hidden_size)

        :param batch_size: we have to know how many elements there are in a batch
        :return: state tuple containing (h, c)
        """
        # LSTM expects hidden to be a tuple=(h, c) : h and c = (num_layers * num_directions, batch, hidden_size)
        h = Variable(torch.zeros(2, batch_size, self.hidden_size))
        c = Variable(torch.zeros(2, batch_size, self.hidden_size))
        if self.cuda_enabled:
            h = h.cuda()
            c = c.cuda()
        return h, c


class Decoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, embedding_dimension, cuda_enabled):
        """
        Initialize the Seq2Seq Decoder.
        NOTE: this module takes only one time step at a time and tries to generate the next token.
        Handle the sequence generation in the complete model iteration

        :param hidden_size: size of the hidden dimension
        :param vocab_size: amount of target classes (words in vocabulary)
        :param embedding_dimension: size of the embedding space
        :param cuda_enabled: flag for using GPU for the model or not
        """
        super(Decoder, self).__init__()

        # init model parameter
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.cuda_enabled = cuda_enabled

        # init layers
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dimension
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dimension,
            hidden_size=hidden_size,
            bidirectional=False
        )
        self.projection = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size
        )

    def forward(self, x, hidden, encoder_output=None):
        """
        Forward pass for one call of this module.
        NOTE: only one time step at a time!

        :param x: the current token in the shape of [batch]
        :param hidden: the dimension of the hidden state
        :param encoder_output: complete output of the encoder - for later use (default = None)
        :return: tuple(output, hidden) of shape: output[batch, vocab], hidden[2, 1, batch, hidden_size]
        """
        # x_input shape: (batch)
        x_embedded = self.embedding(x)  # shape: (batch, embedding)
        x_embedded = x_embedded.unsqueeze(0)  # shape: (1, batch, embedding)
        output, hidden = self.lstm(x_embedded, hidden)

        output = output.squeeze(0)  # shape: (batch, hidden)
        output = F.log_softmax(self.projection(output), dim=-1)  # shape: (batch, vocab)
        return output, hidden
