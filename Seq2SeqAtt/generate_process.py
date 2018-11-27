from Preprocessing.Dataset.JSON.dataset import JsonDataset as Dataset
from Model.seq2seqAtt_model import AttentionModel as Model
import configparser, os, sys, socket
import tensorflow as tf
import numpy as np

SHARE_FILE_PATH = "/tmp/response"
INCOMING_SIGNAL = "<INC>"
OUTGOING_SIGNAL = "<DONE>"
FAIL_SIGNAL = "<FAIL>"

class Generator:
    def __init__(self, dataset, nnmodel, checkpoint_path):
        self.dataset = dataset
        self.nnmodel = nnmodel
        self.nnmodel.build_graph()
        # use CPU for generation
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def build_x(self, json):
        tokens = self.dataset.indexize_text(json)
        tokens = self.dataset.pad(tokens, self.nnmodel.input_size)
        batch = [tokens] * self.nnmodel.batch_size
        x = batch
        y = np.zeros([self.nnmodel.batch_size, self.nnmodel.output_size])
        mask = np.zeros([self.nnmodel.batch_size, self.nnmodel.output_size])
        return (x, y, mask)

    def generate(self, json):
        # TODO: this is not optimal --- tf expects the batch size which it used to train on
        # TODO: also, it expects all of the other stuff which is unknown at inference ....
        x,y,mask = self.build_x(json)
        feed_dict = {
            self.nnmodel.x: x,
            self.nnmodel.y: y,
            self.nnmodel.y_masks: mask,
            self.nnmodel.is_training: False
        }
        print("building result")
        [result] = self.session.run([self.nnmodel.pred_argmax], feed_dict=feed_dict)
        print("done building result")
        # get the first batch
        result = result[0]
        words = [self.dataset.i2w[index] for index in result if index != 0]
        return "".join(words)

def main():
    print(sys.argv)
    assert len(sys.argv) == 4
    # vocab & index mappings
    preprocess_credential_path = sys.argv[1]
    # config for the trained model
    experiment_config_path = sys.argv[2]
    # checkpoint of the trained model
    checkpoint_path = sys.argv[3]

    # we only need the path to the vocab & index mapping here
    dataset = Dataset(dataset_path=None,
                      output_path=preprocess_credential_path,
                      dump_path=None)
    print("load models ...")
    dataset.load()

    experiment_config = configparser.RawConfigParser()
    experiment_config.read(experiment_config_path)

    model = Model(input_size=experiment_config.getint("Model", "input_size"),
                  output_size=experiment_config.getint("Model", "output_size"),
                  batch_size=experiment_config.getint("Model", "batch_size"),
                  model=dataset,
                  embed_size=experiment_config.getint("Embeddings", "hidden_size"),
                  enc_hidden_size=experiment_config.getint("Model", "hidden_size"),
                  dec_hidden_size=experiment_config.getint("Model", "hidden_size") * 2,
                  lr=experiment_config.getfloat("Model", "learning_rate"))

    generator = Generator(dataset, model, checkpoint_path)
    print("Done loading model, awaiting messages")
    soc = socket.socket()
    host = "localhost"
    port = 8912
    soc.bind((host, port))
    soc.listen(5)
    file_comm(soc, generator)
    print("terminating script")

def validate_json(json_string):
    # TODO: validate here
    return True

def read_message(conn):
    length_of_message = int.from_bytes(conn.recv(2), byteorder='big')
    msg = conn.recv(length_of_message).decode("UTF-8")
    return msg

def send_message(conn, msg):
    conn.send(len(msg).to_bytes(2, byteorder='big'))
    conn.send(str.encode(msg + "\n", "UTF-8"))

def read_from_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    return "".join(lines).strip()

def write_to_file(path, text):
    with open(path, "w") as f:
        f.write(text)

def file_comm(soc, generator):
    # wait for java signal
    while True:
        conn, addr = soc.accept()
        length_of_message = int.from_bytes(conn.recv(2), byteorder='big')
        signal = conn.recv(length_of_message).decode("UTF-8")
        print(signal)
        print(INCOMING_SIGNAL)
        if signal == INCOMING_SIGNAL:
            # read message from file
            msg = read_from_file(SHARE_FILE_PATH)
            print(msg)
            if not validate_json(msg):
                print("message not valid!")
                send_message(conn, FAIL_SIGNAL)
                continue
            # generate response
            print("generating response")
            try:
                result = generator.generate(msg)
            except Exception as e:
                send_message(conn, FAIL_SIGNAL)
                print("something went wrong!")
                print(e)
                continue
            # write response to file
            print(result)
            write_to_file(SHARE_FILE_PATH, result)
            # send ready signal to java
            send_message(conn, OUTGOING_SIGNAL)

if __name__ == "__main__":
    main()

