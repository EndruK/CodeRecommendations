import torch.utils.data as data

class Partition(data.Dataset):
    def __init__(self, name, dataframe, index_list):
        """
        constructor - set dataframe and index list
        :param name: name of parition
        :param dataframe: pandas dataframe of all tuples in the coprus
        :param index_list: index list for this partition
        """
        super(Partition, self).__init__()

        self.name = name
        self.dataframe = dataframe
        self.index_list = index_list

        self.partition_size = len(index_list)


    def __len__(self):
        """
        get the length of the partition
        :return: int length
        """
        return self.partition_size

    def __getitem__(self, item):
        """
        get an item out of the dataframe
        :param item: index in partition refers to some shuffled index in the dataframe
        :return: tuple of x and y
        """
        index = self.index_list[item]
        return self.dataframe[index]
