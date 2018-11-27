import json, random, multiprocessing, os, logging as log

TARGET = "expression"
REPLACE = "{<INV>:<EMPTY>}"

def build_dataset(corpus, process_count, destination, statement_limit=1000, file_max_limit=10000):
    """
    Build a Dataset for learning to generate Expressions during programming
    This Dataset has to be processed further in order to let a neural network train on it
    :param corpus: corpus object holding corpus variables
    :param process_count: how many processes should be spawned to build the dataset?
    :param destination: path where the dataset files should be stored
    :return:
    """
    if not os.path.isdir(destination):
        os.makedirs(destination)
    log.info("start building dataset")

    # get the amount of files one process should handle
    size = len(corpus.corpus_file_list) // process_count
    remainder = len(corpus.corpus_file_list) % process_count
    distribution = [size] * process_count
    # distribute the remainder
    for i in range(remainder):
        distribution[i] += 1

    jobs = []
    # create a process pool
    pool = multiprocessing.Pool(processes=process_count)
    for process_id in range(process_count):
        start = sum(distribution[:process_id])
        end = start + distribution[process_id]
        if process_id == process_count - 1:
            process_file_list = corpus.corpus_file_list[start:]
        else:
            process_file_list = corpus.corpus_file_list[start:end]
        # spawn process
        p = pool.apply_async(process_corpus_file_parallel, args=(process_id,
                                                                 process_file_list,
                                                                 destination,
                                                                 statement_limit,
                                                                 file_max_limit))
        jobs.append(p)
    # wait for all jobs to finish
    for job in jobs:
        _ = job.get()
    log.info("done building dataset")


def process_corpus_file_parallel(process_id, process_file_list, destination, statement_limit, file_max_limit):
    """
    function to be called on processes to extract statements and build tuples
    :param process_id:
    :param process_file_list:
    :param destination:
    :return:
    """
    cnt = 0
    file_cnt = 0
    dump_file_name = "process.%d.num.%d" % (process_id, file_cnt)
    file_path = os.path.join(destination, dump_file_name)
    file_x = open(file_path + ".x", "a")
    file_y = open(file_path + ".y", "a")
    log.debug("start process %d" % process_id)
    for i in range(len(process_file_list)):
        log.debug("process #%d (file index %d to %d)" % (process_id, i, len(process_file_list)))
        tuple_generator = extract_expression_statements(process_file_list[i], statement_limit)

        for source, target in tuple_generator:
            file_x.write(source + "\n")
            file_y.write(target + "\n")
            cnt += 1
            if cnt == file_max_limit:
                file_x.close()
                file_y.close()
                file_cnt += 1
                dump_file_name = "process.%d.num.%d" % (process_id, file_cnt)
                file_path = os.path.join(destination, dump_file_name)
                file_x = open(file_path + ".x", "a")
                file_y = open(file_path + ".y", "a")
    log.debug("process %d done" % process_id)
    return True



def extract_expression_statements(file_path, limit=100):
    """
    extract statements from a given method as a generator
    :param file_path:
    :param limit:
    :return:
    """
    with open(file_path, "r") as f:
        text = f.read()
    # json string to json object
    json_object = json.loads(text)
    # get all statements out of the json object
    statements = extract_target_nodes(json_object, TARGET, [])
    # we don't want to use every statement, just sample it
    end = limit if len(statements) > limit else len(statements)
    # also shuffle the statements for sampling
    random.shuffle(statements)
    for i in range(0, end):
        # object -> string
        json_str = str(json_object)
        stmt_str = str(statements[i])
        # replace the extracted statement from source with the replace string
        source = json_str.replace(stmt_str, REPLACE)
        target = stmt_str
        # yield until limit reached
        yield source, target


def extract_target_nodes(node, target, target_list):
    """
    extracts all nodes of type target out of object recursively
    :param node: json tree
    :param target: node type
    :return: list of all subtrees in object with target as root
    """
    # if we have a list, get it and iterate over all list elements
    if isinstance(node, list):
        for item in node:
            extract_target_nodes(item, target, target_list)
    # ignore, if we have a string
    elif isinstance(node, str):
        pass
    # if we have a dict, check all keys for the target
    elif isinstance(node, dict):
        for key, value in node.items():
            if key == target:
                target_list.append(node)
            # expand its children
            target_list = extract_target_nodes(value, target, target_list)
    else:
        raise NotImplementedError("this case is not implemented yet! type: %s" % str(type(node)))
    return target_list
