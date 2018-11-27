import os, sys, random, math, nltk, pickle as pkl, datetime, numpy as np, json
import multiprocessing
# TODO: include logging

def process_vocab_part(file_list, process):
    _process_vocab = {}
    cnt = 0
    for file in file_list:
        with open(file, "r") as f:
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
                 output_path,
                 dump_path):
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

        self.process_count = 25


        self.input_size = 2000
        self.output_size = 200

        self.skip_size = 5

        self.dump_path = dump_path

    def extract_statements(self, file_path, limit=100):
        text = self.read_file(file_path)
        statements = self.slice_method_statements(text)
        # build pairs
        if len(statements) > limit:
            end = limit
        else:
            end = len(statements)
        random.shuffle(statements)
        for statement in statements[:end]:
            json_str = str(json.loads(text))
            stmt_str = str(statement)
            source = json_str.replace(stmt_str, "{<INV>:<EMPTY>}")
            target = stmt_str
            yield source, target

    def slice_method_statements(self, text):
        json_data = json.loads(text)
        statements = self.process_node(json_data, target="expression")
        return statements

    def process_node(self, node, target, target_list=[]):
        if isinstance(node, list):
            for item in node:
                self.process_node(item, target, target_list)
        elif isinstance(node, str):
            pass
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == target:
                    target_list.append(node)
                target_list = self.process_node(value, target, target_list)
        else:
            raise NotImplementedError("tree case not implemented:", type(node))
        return target_list

    def create(self,
                   shuffle=False,
                   word_threshold=5):
        # get all files in the dataset
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".ast"):
                    self.file_paths.append(os.path.join(root, file))
        if shuffle:
            random.shuffle(self.file_paths)
        self.split_dataset()
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
        new_tokens = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token.startswith("'") and len(token) > 1:
                new_tokens.append(token[0])
                new_tokens.append(token[1:])
            else:
                new_tokens.append(token)
        tokens = new_tokens
        del new_tokens
        for token in tokens:
            if token not in self.vocab:
                result.append(self.w2i[self.UNK])
            else:
                result.append(self.w2i[token])
        return result

    def embedding_generator(self, batch_size=32):
        assert self.training_files
        random.shuffle(self.training_files)
        batch_x = []
        batch_y = []
        cnt = 0
        for i in range(len(self.training_files)):
            file = self.training_files[i]
            text = self.read_file(file)
            tokens = self.indexize_text(text)
            skips = self.build_skipgrams(tokens)
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


    def batch_generator(self, dataset, batch_size=32, size=1.0):
        random.shuffle(dataset)
        batch_x, batch_y, batch_y_mask = [], [], []
        cnt = 0
        for i in range(int(len(dataset)*size)):
            file_path = dataset[i]
            for x_text, y_text in self.extract_statements(file_path):
                #print(x_text)
                #print(nltk.word_tokenize(x_text))
                x_tokens = self.indexize_text(x_text)
                y_tokens = self.indexize_text(y_text)
                if len(x_tokens) > self.input_size or len(y_tokens) > self.output_size:
                    continue
                y_mask = [1.0] * len(y_tokens) + [0.0] * (self.output_size-len(y_tokens))
                batch_x.append(self.pad(x_tokens, self.input_size))
                batch_y.append(self.pad(y_tokens, self.output_size))
                batch_y_mask.append(y_mask)
                cnt += 1
                if cnt % batch_size == 0 and cnt > 0:
                    yield batch_x, batch_y, batch_y_mask
                    batch_x, batch_y, batch_y_mask = [], [], []

    def pad(self, tokens, length):
        result = [self.w2i[self.PAD]] * length
        for i in range(len(tokens)):
            result[i] = tokens[i]
        return result

    # this is for exporting
    def pre_build_dataset_pairs(self, dataset, name, processid=0, file_size=int(5e3)):
        print("start building dataset")
        dataset_generator = self.batch_generator(dataset, batch_size=1)
        data = []
        file_cnt = 0
        item_cnt = 0
        for [x], [y], [mask] in dataset_generator:
            data.append([x, y, mask])
            item_cnt += 1
            if item_cnt % 500 == 0:
                print(processid, ":", item_cnt)
            if len(data) == file_size:
                # export it
                filename = os.path.join(self.dump_path, name + "." + "process" + str(processid) + ".pairs." + str(file_cnt))
                with open(filename, "wb") as f:
                    pkl.dump(data, f)
                # clear array
                data = []
                file_cnt += 1

    def pre_build_pair_batch_generator(self, name, batch_size=32, size=1.0):
        # first get all files for the given name
        print("building " + name + " batch generator")
        filelist = []
        for file in os.listdir(self.dump_path):
            if file.startswith(name):
                filelist.append(os.path.join(self.dump_path, file))
        random.shuffle(filelist)
        dataset_size = 0
        # print("begin counting " + name + " samples")
        # for file in filelist:
        #     with open(file, "rb") as f:
        #         tmp = pkl.load(f)
        #     dataset_size += len(tmp)
        #     del tmp
        # print("done counting " + name + " samples")
        cnt = 0
        batch_x, batch_y, batch_y_masks = [], [], []
        for file in filelist:
            with open(file, "rb") as f:
                data = pkl.load(f)
            random.shuffle(data)
            for x, y, mask in data:
                batch_x.append(x)
                batch_y.append(y)
                batch_y_masks.append(mask)
                cnt += 1
                if cnt == int(dataset_size * size):
                    return
                if len(batch_x) == batch_size:
                    yield batch_x, batch_y, batch_y_masks
                    batch_x, batch_y, batch_y_masks = [], [], []