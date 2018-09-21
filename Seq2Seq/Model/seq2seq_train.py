from .seq2seq_model import LSTMModel
import tensorflow as tf
from datetime import datetime
import os
import numpy as np
from tensorflow.python import debug as tf_debug
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from tf.batch_queue import BatchQueue

class Train:
    """
    class to run a training
    """

    def __init__(self,
                 datamodel,
                 embedding_model,
                 logs_path,
                 checkpoint_path,
                 epochs,
                 hidden_size,
                 learning_rate,
                 validation_interval,
                 gpu,
                 print_interval,
                 batch_size):
        self.datamodel = datamodel
        self.embedding_model = embedding_model
        self.enc_hidden = hidden_size
        self.dec_hidden = hidden_size*2
        self.learning_rate = learning_rate
        self.logs_path = logs_path
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.validation_interval = validation_interval
        self.print_interval = print_interval
        self.batch_size = batch_size
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.tf_model = LSTMModel(input_size=datamodel.input_size,
                                  output_size=datamodel.output_size,
                                  batch_size=self.batch_size,
                                  model=datamodel,
                                  embed_size=embedding_model.embedding_size,
                                  enc_hidden=self.enc_hidden,
                                  dec_hidden=self.dec_hidden,
                                  lr=self.learning_rate)
        self.tf_model.build_graph()
        self.render_cnt = 0

    def train(self):
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        now = datetime.now()
        directory = os.path.join(self.logs_path, str(now))
        writer = tf.summary.FileWriter(directory)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            #session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.run(tf.global_variables_initializer())
            embedding_feed = {
                self.tf_model.embedding_placeholder: self.embedding_model.final_embeddings}
            session.run(self.tf_model.embedding_init, feed_dict=embedding_feed)
            global_step = 0
            total_loss = 0
            highest_acc = 0
            last_acc = None
            writer.add_graph(session.graph)
            print("start training")
            for epoch in range(self.epochs):
                train_acc_sum = 0
                training_sampler = self.datamodel.batch_generator(dataset=self.datamodel.training_files,
                                                                  batch_size=self.batch_size)
                # batch_sampler = self.sampler.get_batch(
                #     self.datamodel.train_samples)
                batch_cnt = 0
                for batch_x, batch_y, y_mask in training_sampler:
                    _, loss, batch_acc = session.run(
                        [self.tf_model.optimizer,
                         self.tf_model.loss,
                         self.tf_model.batch_accuracy],
                        feed_dict={
                            self.tf_model.x: batch_x,
                            self.tf_model.y: batch_y,
                            self.tf_model.y_masks: y_mask
                        }
                    )
                    train_summ = tf.Summary()
                    train_summ.value.add(
                        tag="training_accuracy", simple_value=batch_acc
                    )
                    train_summ.value.add(
                        tag="training_loss", simple_value=loss
                    )
                    writer.add_summary(train_summ, global_step)
                    writer.flush()
                    train_acc_sum += batch_acc
                    if global_step % self.print_interval == 0 and global_step > 0:
                        print("global step",
                              str(global_step),
                              "epoch",
                              str(epoch),
                              "batch",
                              str(batch_cnt))
                        print("mean acc over",self.print_interval,"steps", train_acc_sum/self.print_interval)
                        train_acc_sum = 0
                    total_loss += loss
                    batch_cnt += 1
                    if global_step % self.validation_interval == 0 and global_step > 0:
                        valid_acc = self.validate(session, global_step, writer)
                        if valid_acc > highest_acc:
                            highest_acc = valid_acc
                            path = os.path.join(self.checkpoint_path, str(now))
                            if not os.path.isdir(path):
                                os.makedirs(path)
                            with open(os.path.join(path, "note"), "w+") as f:
                                writestring = "Validation Accuracy of checkpoint:\n"
                                writestring += str(highest_acc)
                                f.write(writestring)
                            savepath = saver.save(session, path)
                    global_step += 1
            self.test(session, writer)

    def validate(self, session, global_step, writer):
        print("start validation")
        validation_sampler = self.datamodel.batch_generator(dataset=self.datamodel.validation_files,
                                                            batch_size=self.batch_size)
        # batch_sampler = self.sampler.get_batch(
        #     self.datamodel.validation_samples)
        batch_cnt = 0
        sum_acc = 0
        render_cnt = 0
        for batch_x, batch_y, y_mask in validation_sampler:
            result, acc = session.run(
                [self.tf_model.inference_argmax,
                 self.tf_model.inference_batch_accuracy],
                feed_dict={
                    self.tf_model.x: batch_x,
                    self.tf_model.y: batch_y,
                    self.tf_model.y_masks: y_mask
                })
            sum_acc += acc
            if batch_cnt % self.print_interval == 0 and batch_cnt > 0:
                o = [self.datamodel.index_to_word[i] for i in result[0]]
                print(o)
                print("batch",str(batch_cnt),"current accuracy:", str(sum_acc/batch_cnt))
                x_sample = [self.datamodel.index_to_word[w] for w in batch_x[0]]
                y_sample = [self.datamodel.index_to_word[w] for w in batch_y[0]]
                pred_sample = [self.datamodel.index_to_word[w] for w in result[0]]
                render_cnt += 1
                # print("x_sample: {}\ny_sample: {}\ny_predicted: {}".format(x_sample, y_sample, pred_sample))
                #print(attention_weights.shape)
            batch_cnt += 1
        if batch_cnt > 0:
            mean_acc = sum_acc / batch_cnt
        else:
            mean_acc = 0
        mean_valid_summ = tf.Summary()
        mean_valid_summ.value.add(
            tag="mean_validation_accuracy", simple_value=mean_acc
        )
        writer.add_summary(
            mean_valid_summ, global_step)
        writer.flush()
        print("done validating")
        print("mean acc on validation", mean_acc)
        return mean_acc

    def test(self, session, writer):
        print("start testing")
        test_sampler = self.datamodel.batch_generator(dataset=self.datamodel.testing_files,
                                                      batch_size=self.batch_size)
        # batch_sampler = self.sampler.get_batch(
        #     self.datamodel.test_samples)
        batch_cnt = 0
        summ_add = 0
        test_summ = tf.Summary()
        acc_sum = 0
        for batch_x, batch_y, y_mask in test_sampler:
            [acc] = session.run(
                [self.tf_model.inference_batch_accuracy],
                feed_dict={
                    self.tf_model.x: batch_x,
                    self.tf_model.y: batch_y,
                    self.tf_model.y_masks: y_mask
                }
            )
            acc_sum += acc
            test_summ.value.add(
                tag="testing_accuracy", simple_value=acc
            )
            if batch_cnt % self.print_interval == 0 and batch_cnt > 0:
                print("batch", str(batch_cnt), "current accuracy:",
                      str(acc_sum / batch_cnt))
            batch_cnt += 1
        acc_sum /= batch_cnt
        print("testing done")
        print("mean accuracy on test set", acc_sum)
