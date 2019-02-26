from Seq2Seq_Pytorch_test.Model.vanilla_seq2seq import Encoder, Decoder
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

class AttentionModel:
    def __init__(self):

        self.loss_function = nn.NLLLoss()

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )
        self.encoder = AttEncoder()
        self.decoder = AttDecoder()
        if self.cuda_enabled:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)


class AttEncoder(Encoder):
    def __init__(self, embedding_layer, hidden_size, vocab_size, embedding_dimension, cuda_enabled):
        super(AttEncoder, self).__init__(embedding_layer, hidden_size, vocab_size, embedding_dimension, cuda_enabled)


class AttDecoder(Decoder):
    def __init__(self, embedding_layer, hidden_size, vocab_size, embedding_dimension, cuda_enabled):
        super(AttDecoder, self).__init__(embedding_layer, hidden_size, vocab_size, embedding_dimension, cuda_enabled)
        self.attention_layer = AttentionLayer(self.hidden_size)

    def forward(self, x, hidden, encoder_output=None):
        x_embedded = self.embedding(x)
        x_embedded = x_embedded.unsqueeze(0)
        attention_context = self.attention_layer(encoder_output, hidden)
        # TODO: check dimensions
        # attention_context = attention_context.unsqeeze()
        attention_context = torch.cat((x_embedded, attention_context), -1)
        output, hidden = self.lstm(attention_context, hidden)
        output = output.squeeze(0)
        output = F.log_softmax(self.projection(output), dim=-1)
        return output, hidden

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, encoder_output, last_decoder_state):
        # TODO: check dimensions here
        # get input sequence length
        input_sequence_length = encoder_output.shape[0]
        batch_size = encoder_output.shape[1]
        attention_energies = Variable(torch.zeros(input_sequence_length, batch_size))
        if self.cuda_enabled:
            attention_energies = attention_energies.cuda()
        for i in range(input_sequence_length):
            attention_energies[i] = self._attention_score(last_decoder_state, encoder_output[i])
        # TODO: check dims
        softmax = torch.softmax(attention_energies, dim=0).unsqueeze(0).unsqueeze(0)
        return softmax

    def _attention_score(self, element, state):
        energy = self.attention(element)
        #TODO: check dimensions here
        #energy = energy.squeeze(0)
        energy = torch.bmm(state, energy)
        return energy
