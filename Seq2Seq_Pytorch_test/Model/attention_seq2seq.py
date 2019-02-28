import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Seq2Seq_Pytorch_test.Model.vanilla_seq2seq import VanillaSeq2Seq

class AttentionSeq2Seq(VanillaSeq2Seq):
    def __init__(self,
                 hidden_size,
                 batch_size,
                 vocab_size,
                 embedding_dimension,
                 cuda_enabled,
                 sos_index,
                 eos_index,
                 attention_mode,
                 teacher_force_probability=0.5,
                 gradient_clipping_limit=5,
                 learning_rate=0.001):
        super(AttentionSeq2Seq, self).__init__(
            hidden_size,
            batch_size,
            vocab_size,
            embedding_dimension,
            cuda_enabled,
            sos_index,
            eos_index,
            teacher_force_probability,
            gradient_clipping_limit,
            learning_rate)
        self.attention_mode = attention_mode
        self.decoder = Decoder(
            self.attention_mode,
            self.embedding,
            self.decoder_hidden_size,
            self.vocab_size,
            self.embedding_dimension,
            self.cuda_enabled
        )
        if self.cuda_enabled:
            self.decoder = self.decoder.cuda()
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)


class Decoder(nn.Module):
    def __init__(self,
                 attention_model,
                 embedding_layer,
                 hidden_size,
                 vocab_size,
                 embedding_dimension,
                 cuda_enabled):
        super(Decoder, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.embedding = embedding_layer
        self.embedding_dimension = embedding_dimension
        self.cuda_enabled = cuda_enabled
        self.n_layers = 2

        self.gru = nn.GRU(
            input_size=self.embedding_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers
        )
        self.concat = nn.Linear(
            in_features=self.hidden_size*2,
            out_features=hidden_size
        )
        self.projection = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size
        )
        self.attention = Attention(
            method=self.attention_model,
            hidden_size=self.hidden_size
        )

    def forward(self, input_step, last_hidden, encoder_output):
        embedded = self.embedding(input_step)
        lstm_output, hidden = self.gru(embedded, last_hidden)
        attention_weights = self.attention(lstm_output, encoder_output)
        context = attention_weights.bmm(encoder_output.transpose(0, 1))
        lstm_output = lstm_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((lstm_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.projection(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


class Attention(nn.Module):
    """
    from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
    """
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attention = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attention = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attention(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attention(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == "general":
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attention_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attention_energies = self.dot_score(hidden, encoder_outputs)
        attention_energies = attention_energies.t()
        return F.softmax(attention_energies, dim=1).unsqueeze(1)
