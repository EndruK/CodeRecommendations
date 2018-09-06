import os, random


class Preprocess:
    """
    this class is to prepare an encoder-decoder dataset
    it takes serialized ASTs and takes out a subtree
    """

    def __init__(self, dataset_path, destination):
        # check if path is absolute or relative
        # if not os.path.isabs(dataset_path):
        # dataset_path = os.path.abspath(dataset_path)
        self.dataset_path = dataset_path
        self.preprocess(self.dataset_path, destination)

    def preprocess(self, files, destination):
        """
        preprocesses the given dataset ASTs
        removes subtrees and stores them separately
        """
        # expects a list of ast files
        for ast in files:
            with open(ast) as f:
                text = f.read()
            lines = text.split("\n")
            lines = [line for line in lines if len(line) > 1]
            maintree, subtree = self.slice_tree(lines)
            self.export_sliced_data(ast, maintree, subtree, destination)

    def export_sliced_data(self, source_file, sliced_tree, slice, destination):
        """
        exports the given sliced tree and the slice to training set
        """
        filename_sliced = os.path.basename(source_file) + ".sliced"
        filename_slice = os.path.basename(source_file) + ".slice"
        full_path_new_file_sliced = os.path.join(destination, filename_sliced)
        full_path_new_file_slice = os.path.join(destination, filename_slice)
        with open(full_path_new_file_sliced, "w") as f:
            for line in sliced_tree:
                f.write(line + "\n")
        with open(full_path_new_file_slice, "w") as f:
            for line in slice:
                f.write(line + "\n")

    def slice_tree(self, stree):
        """
        slice the given serialized tree
        """
        # get a random line
        lines = len(stree)
        # keep in mind, we dont want to slice at the first and last node
        random_line = random.randint(1, lines - 2)

        # get the indentation of the selected line
        base_indent = self.get_indent(stree[random_line])
        subtree = []
        subtree.append(stree[random_line])
        end_line = None
        # move over the next lines until we are either at end or back at base_indent
        for line_num in range(random_line + 1, lines):
            # get the indent of that line
            indent = self.get_indent(stree[line_num])
            if indent <= base_indent:
                end_line = line_num
                break
            subtree.append(stree[line_num])
        # get the base tree to the slice position
        base_tree = stree[:random_line]
        # append the invocation token at this position
        base_tree.append("<INV>")
        if end_line is not None:
            # append the remaining tree
            base_tree.extend(stree[end_line:])
        if len(subtree) <= 0:
            base_tree, subtree = self.slice_tree(stree)

        # we have to reset the indentation of the subtree to be rooted at the first prediction
        subtree = self.reset_indentation(subtree)

        return base_tree, subtree

    def reset_indentation(self, subtree):
        """
        resets the indentation of a given tree to start at 0
        """
        result = []
        # get first indentation
        root = subtree[0]
        # set this as offset
        indent_offset = self.get_indent(root)
        for i in range(0, len(subtree)):
            line = subtree[i]
            indent = self.get_indent(line)
            new_indent = indent - indent_offset
            indent_string = "w" + str(new_indent)
            # build new node string
            new_line = indent_string + " " + " ".join(line.split(" ")[1:])
            result.append(new_line)
        return result

    def get_indent(self, line):
        """
        returns the indent of a given serialized AST node (YAML style) as integer
        """
        return int(line.split(" ")[0][1:])