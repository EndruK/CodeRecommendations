import torch.utils.data as data


class Partition(data.Dataset):
    def __init__(self, x, y):
        """
        Initializes the dataset with given symmetric arrays
        :param x: x-array created by sklearn train_test split
        :param y: y-array created by sklearn train_test split
        """
        super(Partition, self).__init__()
        assert(len(x) == len(y))
        self.length = len(x)
        self.x = x
        self.y = y

    def __len__(self):
        """
        get the length of the partition
        :return: int length
        """
        return self.length

    def __getitem__(self, item):
        """
        get an item out of the partition
        :param item: index to the desired tuple
        :return: tuple of x and y
        """
        x = self.x[item]
        y = self.y[item]
        return x, y
