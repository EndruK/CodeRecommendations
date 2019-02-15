import logging as log
from Helper.logging import init_logging
from Seq2Seq_Pytorch_test.Data.dataset import Dataset
from Seq2Seq_Pytorch_test.Model.vanilla_seq2seq import VanillaSeq2Seq
import argparse
import torch.utils.data as data
import time
import datetime
import numpy as np
import os
from tensorboard_logger import configure, log_value
import random
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("csv_path", type=str, help="Path to the CSV holding all tuples")
    parser.add_argument("tokenizer", type=str, help="tokenizer to use on training (nltk, json)")

    parser.add_argument("vocab_top_k", type=int, help="Amount of top k words to keep in vocab")
    parser.add_argument("vocab_build_processes", type=int, help="num of parallel processes for vocab creation")

    parser.add_argument("vocab_export_path", type=str, help="Path to dump the extracted vocab to")
    parser.add_argument("vocab_name", type=str, help="name of the resulting vocab files")

    parser.add_argument("log_level", type=str, help="Define the log level (debug, info, warn, critical)")
    parser.add_argument("log_path", type=str, help="Define the path to store the log file to")

    parser.add_argument("-g", "--gpu_ids", nargs='+', help="Set list of GPUs (default=0)", default=['0'])

    args = parser.parse_args()
    csv_path = args.csv_path
    tokenizer = args.tokenizer

    top_k = args.vocab_top_k
    num_processes = args.vocab_build_processes

    vocab_path = args.vocab_export_path
    vocab_name = args.vocab_name

    log_level = args.log_level
    log_path = args.log_path
    init_logging(log_level, log_path)

    gpu_id_list = args.gpu_ids
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_id_list])
    log.debug("set visible GPUs: %s" % str(gpu_id_list))

    if tokenizer == "nltk":
        from Seq2Seq_Pytorch_test.Data.tokenizers.nltk_tokenizer import NLTKTokenizer as Tok
    elif tokenizer == "json":
        from Seq2Seq_Pytorch_test.Data.tokenizers.json_tokenizer import JsonTokenizer as Tok
    else:
        raise ValueError("wrong tokenizer specified: %s" % tokenizer)
    # initialize the dataset
    dataset = Dataset(csv_path, Tok)
    dataset.split_dataset()
    dataset.build_vocab_single_process(top_k=top_k)
    dataset.dump_vocab(vocab_path, vocab_name)
    # now we can use the vocab for training

    # TODO: parameterize this
    hidden_size = 128
    batch_size = 4
    vocab_size = len(dataset.vocab)
    embedding_dimension = 64
    cuda_enabled = True
    epochs = 50
    validate_every_batch = 2000
    print_every_batch = 100
    model_save_path = os.path.join(args.vocab_export_path, "model")
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_log_dir = os.path.join(args.vocab_export_path, "tensorboard")
    if not os.path.isdir(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    tensorboard_log_dir = os.path.join(tensorboard_log_dir,
                                       datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    if not os.path.isdir(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    configure(tensorboard_log_dir)

    # build pytorch model
    model = VanillaSeq2Seq(
        hidden_size=hidden_size,
        batch_size=batch_size,
        vocab_size=vocab_size,
        embedding_dimension=embedding_dimension,
        cuda_enabled=cuda_enabled,
        sos_index=dataset.word_2_index["SOS"]
    )

    shuffle_data_loader = True
    num_workers = 6

    training_generator = data.DataLoader(dataset.partitions["training"],
                                         batch_size=batch_size,
                                         shuffle=shuffle_data_loader,
                                         num_workers=num_workers,
                                         collate_fn=dataset.partitions["training"].collate)
    validation_generator = data.DataLoader(dataset.partitions["validation"],
                                           batch_size=batch_size,
                                           shuffle=shuffle_data_loader,
                                           num_workers=num_workers,
                                           collate_fn=dataset.partitions["validation"].collate)
    testing_generator = data.DataLoader(dataset.partitions["testing"],
                                        batch_size=batch_size,
                                        shuffle=shuffle_data_loader,
                                        num_workers=num_workers,
                                        collate_fn=dataset.partitions["testing"].collate)
    global_step = 0
    mean_loss = 0
    mean_acc = 0
    validation_cnt = 0
    global_acc = 0
    time_array = []

    log_msg = "[meta]\t"
    log_msg += "[training size]: %d\t" % len(dataset.partitions["training"])
    log_msg += "[validation size]: %d\t" % len(dataset.partitions["validation"])
    log_msg += "[testing size]: %d" % len(dataset.partitions["testing"])
    log.debug(log_msg)

    log.info("[training]\t[start]")
    for i in range(epochs):
        b_cnt = 0
        # iterate over all items in training and bundle mini_batches in random order
        for batch in training_generator:
            start_batch = time.time()
            if len(batch) != batch_size:
                log_msg = "[training]\tbatch is of wrong size - skipping! "
                log_msg += "(expected: %d - got: %d)" % (batch_size, len(batch))
                log.warning(log_msg)
                continue
            try:
                loss, acc = model.training_iteration(batch)
            except Exception as e:
                log.error("There was an error during training!")
                _x = np.array(batch[:, 0].tolist())
                _y = np.array(batch[:, 1].tolist())
                log.debug("x_len: %d" % len(_x))
                log.debug("y_len: %d" % len(_y))
                log.error(str(e))
                raise e
            log_value("training batch loss", loss, global_step)
            log_value("training batch acc", acc, global_step)
            end_batch = time.time()
            time_array.append(end_batch-start_batch)
            mean_loss += loss
            mean_acc += acc
            global_step += 1
            b_cnt += 1
            if global_step % print_every_batch == 0 and global_step > 0:
                mean_it_duration = datetime.timedelta(seconds=float(np.mean(np.array(time_array))))
                log_message = "[training]\t"
                log_message += "[global step]: %d\t" % global_step
                log_message += "[epoch]: %d\t" % i
                log_message += "[batch]: %d\t" % b_cnt
                log_message += "[mean values over %d batches]: " % print_every_batch
                log_message += "loss: %2.4f " % (mean_loss / print_every_batch)
                log_message += "acc: %2.4f\t" % (mean_acc / print_every_batch)
                log_message += "[mean iteration duration]: %s" % mean_it_duration
                log.info(log_message)
                mean_loss = 0
                mean_acc = 0
                time_array = []
            if global_step % validate_every_batch == 0 and global_step > 0:
                log.info("[validation]\t[start]")
                validation_cnt += 1
                valid_step = 0
                valid_loss = 0
                valid_acc = 0
                complete_validation_loss = 0
                complete_validation_acc = 0
                valid_time_array = []
                valid_start_time = time.time()
                for valid_batch in validation_generator:
                    valid_iteration_start = time.time()
                    # NOTE: unexpected batch sizes have to be skipped!
                    if len(valid_batch) != batch_size:
                        log_msg = "[validation]\tbatch is of wrong size - skipping! "
                        log_msg += "(expected: %d - got: %d)" % (batch_size, len(valid_batch))
                        log.warning(log_msg)
                        continue
                    try:
                        loss, acc = model.validation_iteration(valid_batch)
                    except Exception as e:
                        log.error("there was an error during validation!")
                        log.debug("valid batch:", valid_batch)
                        raise e
                    valid_iteration_end = time.time()
                    valid_time_array.append(valid_iteration_end-valid_iteration_start)
                    valid_loss += loss
                    valid_acc += acc
                    complete_validation_loss += loss
                    complete_validation_acc += acc
                    valid_step += 1
                    if valid_step % print_every_batch == 0 and valid_step > 0:
                        mean_it_duration = datetime.timedelta(seconds=float(np.mean(np.array(valid_time_array))))
                        log_message = "[validation]\t"
                        log_message += "[valid batch]: %d\t" % valid_step
                        log_message += "[mean values over %d batches]: " % print_every_batch
                        log_message += "loss: %2.4f " % (valid_loss / print_every_batch)
                        log_message += "acc: %2.4f\t" % (valid_acc / print_every_batch)
                        log_message += "[mean iteration duration]: %s" % mean_it_duration
                        log.info(log_message)
                        valid_loss = 0
                        valid_acc = 0
                        valid_time_array = []
                valid_end_time = time.time()
                log.info("[validation]\t[duration]: %s" % datetime.timedelta(seconds=valid_end_time-valid_start_time))
                complete_validation_loss /= valid_step
                complete_validation_acc /= valid_step
                log_value("validation loss", complete_validation_loss, global_step)
                log_value("validation acc", complete_validation_acc, global_step)
                log_message = "[validation]\t"
                log_message += "[results #%d]\t" % validation_cnt
                log_message += "[mean values over complete validation]: "
                log_message += "loss: %2.4f " % complete_validation_loss
                log_message += "acc: %2.4f\t" % complete_validation_acc
                log.info(log_message)
                if complete_validation_acc > global_acc:
                    log.info("Better accuracy reached! old: %2.4f new: %2.4f - Saving model"
                             % (global_acc, complete_validation_acc))
                    model.save(path=model_save_path, name="best.checkpoint", acc=complete_validation_acc)
                    global_acc = complete_validation_acc
                rnd_index = random.randint(0, len(dataset.partitions["validation"]))
                rnd_x, rnd_y = dataset.partitions["validation"][rnd_index]
                rnd_x_indices = dataset.partitions["validation"].collate_single(rnd_x)
                rnd_result = model.generation_iteration(rnd_x_indices)
                result_string = "".join(dataset.index_array_to_text(rnd_result))
                log_str = "[validation]\t[example generation]\n"
                log_str += "[x]: %s\n" % rnd_x
                log_str += "[target]: %s\n" % rnd_y
                log_str += "[result]: %s\n" % result_string
                log.info(log_str)
    log.info("[training]\t[done]")
    log.info("[testing]\t[start]")
    test_loss = []
    test_acc = []
    print_loss = 0
    print_acc = 0
    test_cnt = 0
    test_start = time.time()
    test_time_array = []

    model.load(path=model_save_path, name="best.checkpoint")

    for batch in testing_generator:
        _s = time.time()
        if len(batch) != batch_size:
            log_msg = "[testing]\tbatch is of wrong size - skipping! "
            log_msg += "(expected: %d - got: %d)" % (batch_size, len(batch))
            log.warning(log_msg)
            continue
        loss, acc = model.validation_iteration(batch)
        _e = time.time()
        test_time_array.append(_e-_s)
        test_loss.append(loss)
        test_acc.append(acc)
        print_loss += loss
        print_acc += acc
        test_cnt += 1
        if test_cnt % print_every_batch == 0 and test_cnt > 0:
            mean_it_duration = datetime.timedelta(seconds=float(np.mean(np.array(test_time_array))))
            log_message = "[testing]\t"
            log_message += "[test batch]. %d\t" % test_cnt
            log_message += "[mean values over %d batches]: " % print_every_batch
            log_message += "loss: %2.4f " % (print_loss / print_every_batch)
            log_message += "acc: %2.4f\t" % (print_acc / print_every_batch)
            log_message += "[mean iteration duration]. %s" % mean_it_duration
            log.info(log_message)
            print_loss = 0
            print_acc = 0
            test_time_array = []
    log_message = "[testing]\t[done]"
    log.info(log_message)
    test_end = time.time()
    log_message = "[testing]\t[duration]: %s" % datetime.timedelta(seconds=test_end-test_start)
    log.info(log_message)
    log_message = "[testing]\t"
    log_message += "[results]\t"
    log_message += "[mean values over complete test set]: "
    log_message += "loss: %2.4f " % np.mean(np.array(test_loss))
    log_message += "acc: %2.4f" % np.mean(np.array(test_acc))
    log.info(log_message)
