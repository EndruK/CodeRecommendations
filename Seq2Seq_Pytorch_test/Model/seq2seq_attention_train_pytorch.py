from Seq2Seq_Pytorch_test.Model.seq2seq_attention_pytorch import AttTrainingHelper
import time, math, logging as log
from tensorboard_logger import configure, log_value

def run_training(
        encoder_time_size,
        decoder_time_size,
        hidden_size,
        vocab_len,
        embedding_dim,
        batch_size,
        learning_rate,
        sos,
        eos,
        epochs,
        validation_interval,
        log_interval,
        data,
        use_cuda,
        tensorboard_log_dir,
        model_store_path
):
    training_helper = AttTrainingHelper(
        encoder_time_size=encoder_time_size,
        decoder_time_size=decoder_time_size,
        hidden_size=hidden_size,
        vocab_len=vocab_len,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sos=sos,
        eos=eos,
        use_cuda=use_cuda
    )
    validation_iterations = 2000  # TODO: magic number
    global_step = 0
    highest_train_acc = 0
    highest_validation_acc = 0
    best_model_name = ""
    # configure tensorboard logdir
    configure(tensorboard_log_dir)

    start_time = time.time()

    print_loss = 0
    print_acc = 0
    log.info("start training")
    # one epoch = complete training set run
    for epoch in range(1, epochs+1):
        log.info("entering epoch #" + str(epoch))
        # build a batch generator
        training_batch_generator = data.pre_build_pair_batch_generator("training")

        for x_training, y_training, _ in training_batch_generator:
            # pass through model and retrieve results
            training_batch_loss,\
                training_batch_accuracy,\
                batch_attention = training_helper.training_iteration(x_training, y_training)

            log_value("training batch loss", training_batch_loss, global_step)
            log_value("training batch acc", training_batch_accuracy, global_step)

            print_loss += training_batch_loss
            print_acc += training_batch_accuracy

            if training_batch_accuracy > highest_train_acc:
                highest_train_acc = training_batch_accuracy
            global_step += 1
            # logging step reached
            if global_step % log_interval == 0:
                print_loss_average = print_loss / log_interval
                print_acc_average = print_acc / log_interval
                print_loss = 0
                print_acc = 0
                print_string = "%s,\tglobal step: %s,\tepoch: %s,\ttrain-loss: %.4f, train-acc: %.4f" \
                               % (time_diff(start_time,epoch / epochs),
                                  str(global_step),
                                  str(epoch),
                                  print_loss_average,
                                  print_acc_average)

                log.info(print_string)
            # validation step reached
            if global_step % validation_interval == 0:
                log.info("start validation")
                valid_step = 1
                full_validation_loss = 0
                full_validation_accuracy = 0
                validation_batch_generator = data.pre_build_pair_batch_generator("validation")
                for x_validation, y_validation,_ in validation_batch_generator:
                    validation_batch_loss, \
                        validation_batch_accuracy, \
                        validation_batch_attention = training_helper.valid_test_iteration(x_validation, y_validation)
                    full_validation_loss += validation_batch_loss
                    full_validation_accuracy += validation_batch_accuracy
                    # logging step reached
                    if valid_step % log_interval == 0:
                        print_string = "%s,\tvalidation\tglobal step: %s, valid step: %s, valid-acc: %.4f" \
                                       % (time_diff(start_time),
                                          str(global_step),
                                          str(valid_step),
                                          full_validation_accuracy/valid_step)
                        log.info(print_string)
                    valid_step += 1
                    # we don't want to run validation forever
                    if valid_step == validation_iterations:
                        break
                mean_validation_accuracy = full_validation_accuracy / valid_step
                mean_validation_loss = full_validation_loss / valid_step
                log_value("validation accuracy", mean_validation_accuracy, global_step)
                if mean_validation_accuracy > highest_validation_acc:
                    highest_validation_acc = mean_validation_accuracy
                    name = "acc" + "%.4f" % mean_validation_accuracy + ".model"
                    log.info("new best accuracy!")
                    log.debug("saving new model \"name = " + name + "\" to " + model_store_path)
                    training_helper.store_model(path=model_store_path, name=name, store_model=True)
                    best_model_name = name
                log.info("done validating model")
                log.info("mean validation accuracy = %.4f" % mean_validation_accuracy)
                log.info("mean validation loss = %.4f" % mean_validation_loss)

            # TODO: deal with no change in accuracy here

    # run test on last version of model
    testing_batch_generator = data.pre_build_pair_batch_generator("testing")
    test_acc = 0
    test_loss = 0
    cnt = 1
    log.info("start test run on best version of model")
    # load the best checkpoint
    training_helper.load_model(checkpoint_path=model_store_path, checkpoint_name=best_model_name)
    # deactivate gradient calculation
    for x_testing, y_testing, _ in testing_batch_generator:
        testing_batch_loss, \
            testing_batch_accuracy , \
            testing_batch_attention = training_helper.valid_test_iteration(x_testing, y_testing)
        test_loss += testing_batch_loss
        test_acc += testing_batch_accuracy
        if cnt % log_interval == 0:
            print_string = "%s,\ttesting\tstep: %s, test-batch-acc: %.4f" \
                           % (time_diff(start_time),
                              str(cnt),
                              test_acc/cnt)
            log.info(print_string)
        cnt += 1
    result_acc = test_acc/(cnt-1)
    result_loss = test_loss/(cnt-1)
    log.info("done testing model")
    log.info("mean test accuracy = %.4f" % result_acc)
    log.info("mean test loss = %.4f" % result_loss)
    log.info("end of training")


def time_to_min(t):
    """
    translates a given time.time() into minutes
    :param t: time.time() stamp
    :return: time.time() stamp in minutes
    """
    m = math.floor(t / 60)
    t -= m * 60
    return "%dm %ds" % (m, t)


def time_diff(first, p=None):
    """
    calculates elapsed time and also how much time left estimated
    :param first: start of loop
    :param p: current epoch / max epoch
    :return: string of both
    """
    now = time.time()
    diff = now - first
    if p is None:
        return "%s" % (time_to_min(diff))
    es = diff / p
    rs = es - diff

    return "%s (%s remaining)" % (time_to_min(diff), time_to_min(rs))
