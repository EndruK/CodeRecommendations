from Seq2Seq_Pytorch_test.Model.vanilla_seq2seq import Encoder, Decoder


class AttEncoder(Encoder):
    def __init__(self):
        super(AttEncoder, self).__init__(None, None, None, None)
        pass


class AttDecoder(Decoder):
    def __init__(self):
        super(AttDecoder, self).__init__(None, None, None, None)
        pass


class AttentionLayer:
    def __init__(self):
        pass
