import os, sys, random, math, nltk, pickle as pkl, datetime, numpy as np
import multiprocessing


def process_vocab_part(file_list, process):
    _process_vocab = {}
    cnt = 0
    for file_x, _ in file_list:
        with open(file_x, "r") as f:
            text = f.read()
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            if token not in _process_vocab:
                _process_vocab[token] = 1
            else:
                _process_vocab[token] += 1
        if cnt % 200 == 0 and cnt > 0:
            print("process {}: file {} of {}".format(process, cnt, len(file_list)))
        cnt += 1
    return _process_vocab

class JsonDataset:
    def __init__(self,
                 dataset_path,
                 output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.file_paths = []
        self.vocab = []
        self.w2i = {}
        self.i2w = {}

        self.UNK = "<UNK>"
        self.PAD = "<PAD>"
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.INV = "<INV>"

        self.process_count = 7


        self.input_size = 2000
        self.output_size = 200

        self.skip_size = 5

    def create(self,
               subset=None,
               shuffle=False,
               word_threshold=5):
        if subset is None:
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith(".x"):
                        x = os.path.join(root, file)
                        y = os.path.join(root, file[:-1]+"y")
                        self.file_paths.append([x, y])
        else:
            assert subset
            with open(subset, "rb") as f:
                self.file_paths = pkl.load(f)
        if shuffle:
            random.shuffle(self.file_paths)
        self.split_dataset()
        # self.build_vocab(word_threshold)
        self.build_vocab_parallel(word_threshold)
        self.export()

    def split_dataset(self):
        scheme = [.7, .2, .1]
        training_length = math.floor(scheme[0] * len(self.file_paths))
        validation_length = math.floor(scheme[1] * len(self.file_paths))
        self.training_files = self.file_paths[: training_length]
        self.validation_files = self.file_paths[training_length: training_length + validation_length]
        self.testing_files = self.file_paths[training_length + validation_length:]

    def read_file(self, path):
        with open(path, "r") as f:
            text = f.read()
        return text

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text)
        return tokens




    def build_vocab_parallel(self, threshold):
        print("build vocab parallel using {} threads".format(self.process_count))
        assert self.training_files
        _vocab = {}
        sublist_length = math.floor(len(self.training_files) / self.process_count)
        _subvocabs = []



        jobs = []
        pool = multiprocessing.Pool(processes=self.process_count)
        for i in range(self.process_count):
            start_index = i * sublist_length
            end_index = (i+1) * sublist_length
            if i == self.process_count-1:
                process_file_list = self.training_files[start_index:]
            else:
                process_file_list = self.training_files[start_index:end_index]
            p = pool.apply_async(process_vocab_part, args=(process_file_list, i))
            jobs.append(p)
        for job in jobs:
            _subvocabs.append(job.get())
        for subvocab in _subvocabs:
            for key, value in subvocab.items():
                if key not in _vocab:
                    _vocab[key] = value
                else:
                    _vocab[key] += value
        sorted_vocab = sorted([[word, freq] for word, freq in _vocab.items()], key=lambda x: x[1], reverse=True)
        cleaned_vocab = [word for word, freq in sorted_vocab if freq >= threshold]
        self.vocab = [self.PAD, self.UNK, self.SOS, self.EOS]
        if self.INV not in cleaned_vocab:
            self.vocab.append(self.INV)
        self.vocab += cleaned_vocab
        for i, word in enumerate(self.vocab):
            self.w2i[word] = i
            self.i2w[i] = word
        print("done building vocab -> length = ", str(len(self.vocab)))


        # for i in range(self.process_count):
        #     start_index = i * sublist_length
        #     end_index = (i+1) * sublist_length
        #     if i == self.process_count-1:
        #         process_file_list = self.training_files[start_index:]
        #     else:
        #         process_file_list = self.training_files[start_index:end_index]
        #     p = multiprocessing.Process(target=self.process_vocab_part, args=(process_file_list,))
        #     jobs.append(p)
        #     p.start()
        # for j in jobs:
        #     j.join()


    def build_vocab(self, threshold):
        print("build vocab")
        assert self.training_files
        _vocab = {}
        cnt = 0
        durations = []
        complete_time = 0
        for file_x, _ in self.training_files:
            start = datetime.datetime.now()
            text = self.read_file(file_x)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in _vocab:
                    _vocab[token] = 1
                else:
                    _vocab[token] += 1
            end = datetime.datetime.now()
            duration = (end - start).total_seconds()
            #print(duration)
            durations.append(duration)
            if cnt % 100 == 0 and cnt > 0:
                sum_duration = np.sum(np.array(durations))
                complete_time += sum_duration
                durations = []
                print("processing file",
                      str(cnt),
                      "of",
                      str(len(self.training_files)),
                      "time taken:",
                      str(sum_duration)+"s",
                      "remaining:",
                      len(self.training_files)/cnt*complete_time/60,
                      "min")
            cnt += 1
        # clean up
        sorted_vocab = sorted([[word, freq] for word, freq in _vocab.items()], key=lambda x: x[1], reverse=True)
        cleaned_vocab = [word for word, freq in sorted_vocab if freq >= threshold]
        self.vocab = [self.PAD, self.UNK, self.SOS, self.EOS]
        if self.INV not in cleaned_vocab:
            self.vocab.append(self.INV)
        self.vocab += cleaned_vocab
        for i, word in enumerate(self.vocab):
            self.w2i[word] = i
            self.i2w[i] = word
        print("done building vocab -> length = ", str(len(self.vocab)))

    def export(self):
        assert self.vocab
        assert self.i2w
        assert self.w2i
        vocab_path = os.path.join(self.output_path, "JSON.vocab")
        w2i_path = os.path.join(self.output_path, "JSON.w2i")
        i2w_path = os.path.join(self.output_path, "JSON.i2w")
        training_paths = os.path.join(self.output_path, "JSON.training_paths")
        validation_paths = os.path.join(self.output_path, "JSON.validation_paths")
        testing_paths = os.path.join(self.output_path, "JSON.testing_paths")
        if self.create_folder(self.output_path):
            print("folder created")
        with open(vocab_path, "wb") as f:
            pkl.dump(self.vocab, f)
        with open(w2i_path, "wb") as f:
            pkl.dump(self.w2i, f)
        with open(i2w_path, "wb") as f:
            pkl.dump(self.i2w, f)
        with open(training_paths, "wb") as f:
            pkl.dump(self.training_files, f)
        with open(validation_paths, "wb") as f:
            pkl.dump(self.validation_files, f)
        with open(testing_paths, "wb") as f:
            pkl.dump(self.testing_files, f)

    def create_folder(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
            return True
        return False

    def load(self):
        vocab_path = os.path.join(self.output_path, "JSON.vocab")
        w2i_path = os.path.join(self.output_path, "JSON.w2i")
        i2w_path = os.path.join(self.output_path, "JSON.i2w")
        training_paths = os.path.join(self.output_path, "JSON.training_paths")
        validation_paths = os.path.join(self.output_path, "JSON.validation_paths")
        testing_paths = os.path.join(self.output_path, "JSON.testing_paths")
        check = os.path.isfile(vocab_path) and \
            os.path.isfile(w2i_path) and \
            os.path.isfile(i2w_path) and \
            os.path.isfile(training_paths) and \
            os.path.isfile(validation_paths) and \
            os.path.isfile(testing_paths)
        if not check:
            print("at least one dataset file cannot be loaded - aborting")
            sys.exit(0)
        with open(vocab_path, "rb") as f:
            self.vocab = pkl.load(f)
        with open(w2i_path, "rb") as f:
            self.w2i = pkl.load(f)
        with open(i2w_path, "rb") as f:
            self.i2w = pkl.load(f)
        with open(training_paths, "rb") as f:
            self.training_files = pkl.load(f)
        with open(validation_paths, "rb") as f:
            self.validation_files = pkl.load(f)
        with open(testing_paths, "rb") as f:
            self.testing_files = pkl.load(f)

    def indexize_text(self, text):
        result = []
        tokens = self.tokenize(text)
        for token in tokens:
            if token in self.vocab:
                result.append(self.w2i[token])
            else:
                result.append(self.w2i[self.UNK])
        return result

    def embedding_generator(self, batch_size=32):
        assert self.training_files
        random.shuffle(self.training_files)
        batch_x = []
        batch_y = []
        cnt = 0
        for i in range(len(self.training_files)):
            x, _ = self.training_files[i]
            x_text = self.read_file(x)
            x_tokens = self.indexize_text(x_text)
            skips = self.build_skipgrams(x_tokens)
            for x, y in skips:
                batch_x.append(x)
                batch_y.append(y)
                cnt += 1
                if cnt % batch_size == 0 and cnt > 0:
                    yield batch_x, batch_y
                    batch_x, batch_y = [], []

    def build_skipgrams(self, tokens):
        result = []
        for i in range(len(tokens)):
            logit = []
            left = i - self.skip_size if i - self.skip_size > 0 else 0
            for j in range(left, i):
                logit.append(tokens[j])
            right = i + self.skip_size if i + self.skip_size < len(tokens)-1 else len(tokens)-1
            for j in range(i+1, right+1):
                logit.append((tokens[j]))
            label = tokens[i]
            for l in logit:
                result.append([l, [label]])
        random.shuffle(result)
        return result

    def batch_generator(self, dataset, batch_size=32):
        random.shuffle(dataset)
        batch = []
        for i in range(len(dataset)):
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = []
            x_path, y_path = dataset[i]
            x_text = self.read_file(x_path)
            y_text = self.read_file(y_path)
            x_tokens = self.indexize_text(x_text)
            y_tokens = self.indexize_text(y_text)
            if len(x_tokens) > self.input_size or len(y_tokens) > self.output_path:
                continue
            batch.append([x_tokens, y_tokens])

    def pad(self, tokens, length):
        result = [self.w2i[self.PAD]] * length
        for i in range(len(tokens)):
            result[i] = tokens[i]
        return result
