import tensorflow as tf
import numpy as np
import datetime
from random import shuffle, randint
import os
import math
from sklearn.manifold import TSNE
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class JSONEmbedding:
    def __init__(self,
                 batch_size,
                 dataset,
                 embedding_size,
                 logs_path,
                 model_checkpoint_path,
                 epochs,
                 gpu):
        # tf.reset_default_graph()
        self.batch_size = batch_size
        self.dataset = dataset
        self.embedding_size = embedding_size
        self.logs_path = logs_path
        self.model_checkpoint_path = model_checkpoint_path
        self.epochs = epochs
        self.print_every = 500
        self.validate_every = 3000
        self.nearest_neighbors = 8
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.num_sampled = 64
        self.valid_window = 100
        self.valid_size = 16
        self.valid_examples = np.random.choice(
            self.valid_window, self.valid_size, replace=False)


    def build_graph(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name="inputs")
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="labels")
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        with tf.device('/gpu:0'):
            self.embeddings = tf.Variable(tf.random_uniform([len(self.dataset.vocab), self.embedding_size], -1.0, 1.0))
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            self.nce_weights = tf.Variable(tf.truncated_normal([len(self.dataset.vocab), self.embedding_size]))
            self.nce_biases = tf.Variable(tf.zeros([len(self.dataset.vocab)]))

        self.loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=self.train_labels,
            inputs=self.embed,
            num_sampled=self.num_sampled,
            num_classes=len(self.dataset.vocab)
        ))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)

        # cosine similarity
        self.norm = tf.sqrt(tf.reduce_mean(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / self.norm
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)

    def train(self):
        print("start the training")
        self.build_graph()
        print("done building graph")
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        now = datetime.datetime.now()
        logs_path = os.path.join(self.logs_path, str(now))
        writer = tf.summary.FileWriter(logs_path)
        global_step = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            session.run(init)
            writer.add_graph(session.graph)
            avg_loss = 0
            loss_array = []
            for epoch in range(self.epochs):
                print("epoch:", epoch)
                batch_generator = self.dataset.embedding_generator(self.batch_size)
                batch_cnt = 0
                # iterate over all batches
                for batch_x, batch_y in batch_generator:
                    batch_cnt += 1
                    _, cur_loss = session.run([self.optimizer, self.loss], feed_dict={
                        self.train_inputs: batch_x,
                        self.train_labels: batch_y
                    })
                    avg_loss += cur_loss
                    if global_step % self.print_every == 0 and global_step > 0:
                        avg_loss /= self.print_every
                        train_summ = tf.Summary()
                        train_summ.value.add(
                            tag="training loss",
                            simple_value=avg_loss
                        )
                        writer.add_summary(train_summ, global_step)
                        writer.flush()
                        print("avg loss at epoch:", epoch, ",step:", batch_cnt, "is:", avg_loss)
                        loss_array.append(avg_loss)
                        avg_loss = 0
                    if global_step % self.validate_every == 0 and global_step > 0:
                        print("start validation")
                        sim = self.similarity.eval()
                        # validation_generator = self.build_batch(self.skip_tuples_validation)
                        for i in range(self.valid_size):
                            valid_word = self.dataset.i2w[self.valid_examples[i]]
                            top_k = self.nearest_neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to ' + valid_word
                            for k in range(top_k):
                                close_word = self.dataset.i2w[nearest[k]]
                                log_str += " ," + str(close_word)
                            print(log_str)
                    global_step += 1
                self.final_embeddings = self.normalized_embeddings.eval()
                save_folder = self.model_checkpoint_path
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                save_path = saver.save(session, os.path.join(save_folder, "embedding.ckpt"))
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000, method='barnes_hut')
                limit = 50
                start = randint(0, len(self.final_embeddings) - limit - 1)
                end = start + limit
                low_dim_embeds = tsne.fit_transform(self.final_embeddings[start:end, :])
                labels = [self.dataset.i2w[i] for i in range(limit)]
                self.plot_vectors(low_dim_embeds, labels, os.path.join(
                    save_folder, "embeddings_plot.png"))
                self.plot_loss(loss_array, os.path.join(
                    save_folder, "embedding_training_loss.png"))

    def load_checkpoint(self, path):
        """
        load a pretrained model from a given path
        """
        print("load pretrained embedding weights")
        self.build_graph()
        print("done building graph")
        saver = tf.train.Saver()
        restore_from = os.path.join(path, "embedding.ckpt")
        with tf.Session() as session:
            saver.restore(session, restore_from)
            self.final_embeddings = self.normalized_embeddings.eval()

    def store_embedding_as_np(self, path):
        """
        save the embedding matrix as numpy dump
        """
        print("save embedding matrix at", path)
        np.savetxt(path, self.final_embeddings)
        print("saved")

    def restore_np_embeddings(self, path):
        """
        restore a dumped embedding matrix from a path
        """
        print("restoring embedding matrix from", path)
        self.final_embeddings = np.loadtxt(path)
        print("restored")

    def plot_loss(self, array, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(array)
        ax.set_xlabel("steps")
        ax.set_ylabel("loss")
        ax.set_title("embedding training loss")
        ax.set_yscale("log")
        plt.savefig(title)

    def plot_vectors(self, low_dim_embeds, labels, filename):
        assert low_dim_embeds.shape[0] >= len(
            labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))
        # plt.scatter(low_dim_embeds[:, 0], low_dim_embeds[:, 1])
        for i, label in enumerate(labels):
            x, y = low_dim_embeds[i, :]
            # x = low_dim_embeds[i, 0]
            # y = low_dim_embeds[i, 1]
            plt.scatter(low_dim_embeds[i, 0], low_dim_embeds[i, 1])
            plt.annotate(label,
                         xy=(x, y),
                         xycoords='data',
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'))
        plt.savefig(filename)
