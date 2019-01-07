from torch.utils import data
import os, logging as log, pickle as pkl, multiprocessing


class Dataset(data.Dataset):
    """
    class to handle pre-built tuples for every partitions (training, validation, testing)
    or even to handle direct input
    """
    def __init__(self, path_to_dump, partition_name, process_count=7):
        """

        :param path_to_dump: path where the preprocessed dataset is
        :param partition_name: name of the partition
        :param process_count: number of processes to count the dataset (default=5)
        """
        super(Dataset, self).__init__()

        self.partition_name = partition_name

        # get all files for partition
        files = []
        log.info("initializing " + partition_name + " dataset")
        log.debug("searching for pre-built tuples in " + path_to_dump)
        for file in os.listdir(path_to_dump):
            if file.startswith(partition_name):
                files.append(os.path.join(path_to_dump, file))
        with open(os.path.join(path_to_dump, "tuple_keys"), "r") as f:
            lines = f.readlines()
        # TODO: work with an index here which is in tuple_keys
        # TODO: change preprocessing to include a mapping



    def __len__(self):
        """
        gives the length of the dataset
        :return:
        """
        return self.size

    def __getitem__(self, index):
        """
        get a tuple for the dataset
        :param index:
        :return: X and Y
        """

        pass


    #
    #
    # # TODO: OBSOLETE
    # def get_partition_size(self, files, process_count):
    #     """
    #     run through the complete pre processed partition and count how many tuples there are
    #     :param files: list containing all paths to all pre-processed files
    #     :param process_count: how many processes should be spawned
    #     :return: size of the partition as int
    #     """
    #     # how many tuples are there in the set?
    #     partition_size = 0
    #     # split files for all processes
    #     num_of_files = len(files) // process_count # files per process
    #     distribution = [num_of_files] * process_count
    #     num_remainder = len(files) % process_count
    #     # add one to distribution to handle all files
    #     for i in range(num_remainder):
    #         distribution[i] += 1
    #
    #     # create a process pool
    #     pool = multiprocessing.Pool(processes=process_count)
    #     jobs = []
    #     for i in range(len(distribution)):
    #         # start is the sum of all previous distribution numbers
    #         start = sum(distribution[:i])
    #         # end is start + current distribution number
    #         end = start + distribution[i]
    #         if i == len(distribution)-1:
    #             end = None
    #         # spawn new process
    #         p = pool.apply_async(Dataset.get_length, args=(i, files, start, end))
    #         # append it to the job container
    #         jobs.append(p)
    #     for job in jobs:
    #         # wait for all processes to return
    #         result = job.get()
    #         partition_size += result
    #     log.info("done getting partition size for %s" % self.partition_name)
    #     log.debug("size = %d" % partition_size)
    #     return partition_size
    #
    # # TODO: OBSOLETE
    # @staticmethod
    # def get_length(process_id, files, start, end):
    #     """
    #     process function to work with a subset of the files
    #     :param process_id: id of the process
    #     :param files: list of all files of the partition
    #     :param start: start index
    #     :param end: end index
    #     :return: length of this subset
    #     """
    #     log.debug("starting process #%d (file index %d to %d)" % (process_id, start, end))
    #     length = 0
    #     if end is not None:
    #         sub_files = files[start:end]
    #     else:
    #         sub_files = files[start:]
    #     for i in range(len(sub_files)):
    #         with open(sub_files[i], "rb") as f:
    #             data = pkl.load(f)
    #         length += len(data)
    #         log.debug("process " + str(process_id) + " file " + str(i) + " of " + str(len(sub_files)))
    #     log.debug("process %d done" % process_id)
    #     return length


# if __name__ == "__main__":
#     log.basicConfig(
#         level=log.DEBUG,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=[
#             log.StreamHandler()
#         ])
#     ds = Dataset("/home/andre/Git/CodeRecommendations/DumpDump/pre_built_dataset", "training")
