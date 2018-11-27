import os

def get_fullpath(directory):
    """
    returns a list containing the full path for all files in a dir
    """
    # one liner is not enough, maybe the folder is nested
    path_list = []
    for f in os.listdir(directory):
        p = os.path.join(directory, f)
        if not os.path.isdir(p):
            path_list.append(p)
        else:
            path_list.extend(get_fullpath(p))
    return path_list


def join_dictionaries(a, b):
    """
    joins two dictionaries together - summing up values
    """
    for key, value in b.items():
        if key in a:
            a[key] += value
        else:
            a[key] = value
    return a
