import os
import random
import logging as log
import json
import csv


class Preprocessor:
    TARGET_AST_NODE = "statements"
    INVOCATION_STRING = '{"INV":"EMPTY"}'
    STATEMENT_LIMIT = 100

    @staticmethod
    def preprocess(path_to_serialized_ast_methods, output_path):
        """
        Preprocessing procesdure
        creates a csv file for all x-y tuples
        :param path_to_serialized_ast_methods: absolute path to serialized AST files of methods
        :param output_path: target path of the resulting csv
        """
        log.debug("start preprocessing")
        # get the paths to all method files
        log.debug("retrieving file list")
        paths = []
        for root, dirs, files in os.walk(path_to_serialized_ast_methods):
            for file in files:
                if file.endswith(".ast"):
                    p = os.path.join(root, file)
                    paths.append(p)
        random.shuffle(paths)

        log.debug("done retrieving file list")
        log.info("amount of files in dataset: %d" % len(paths))

        # open csv file
        csv_file_path = os.path.join(output_path, "tuples.csv")
        # check if there is already a file
        if os.path.isfile(csv_file_path):
            log.debug("removing old file at %s" % csv_file_path)
            os.remove(csv_file_path)
        open(csv_file_path, "w").close()

        csv_file = open(csv_file_path, "w")
        csv_writer = csv.writer(csv_file, lineterminator="\n", delimiter=",", quotechar="'")
        csv_writer.writerow(["x", "y"])

        # extracting target nodes
        log.debug("starting statement extraction")
        sum_of_tuples = 0
        for i in range(len(paths)):
            if i % 100 == 0:
                log.debug("processing statement extraction - file %d of %d" % (i, len(paths)))
            # load the json of the current file
            with open(paths[i], "r") as f:
                json_object = json.load(f)
            tuples = Preprocessor.extract_statements(json_object, 500)
            sum_of_tuples += len(tuples)
            for (x, y) in tuples:
                csv_writer.writerow((x, y))
        csv_file.close()
        log.info("amount of tuples over whole dataset: %d" % sum_of_tuples)
        log.debug("done statement extraction")
        log.debug("done preprocessing")
        log.info("path to csv file: %s" % csv_file_path)

    @staticmethod
    def extract_statements(json_object, limit=100):
        """
        Extracts all statements of a method based on Target Node Type.
        shuffles statements and takes only max. 100 statements
        :param json_object: json object of a method
        :param limit: how many statements should be extracted (default=100)
        :return: list of tuples containing (x,y)
        """
        statement_lists = Preprocessor.process_json_node(json_object,
                                                         target=Preprocessor.TARGET_AST_NODE,
                                                         target_list=[])
        statements = []
        for statement_list in statement_lists:
            for statement in statement_list:
                statements.append(statement)
        # try to prevent that nasty json bug where the old json objects are still in memory
        del statement_lists
        result = []
        count = 0
        random.shuffle(statements)
        json_string = json.dumps(json_object)
        for statement in statements:
            if limit != -1 and count >= limit:
                break
            # Workaround for that nasty json bug - just move on if statement can't be found in json string
            found_target_in_json = json_string.find(json.dumps(statement))
            if found_target_in_json == -1:
                continue
            x = json_string
            y = json.dumps(statement)
            x = x.replace(y, Preprocessor.INVOCATION_STRING)
            result.append((x, y))
            count += 1
        return result

    @staticmethod
    def process_json_node(node, target, target_list):
        """
        recursive function to extract all nodes of a given type
        :param node: current root node
        :param target: node type which should be found
        :param target_list: list where all target nodes are appended
        :return: target_list
        """
        if isinstance(node, list):
            for item in node:
                target_list = Preprocessor.process_json_node(item, target, target_list)
        elif isinstance(node, str):
            pass
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == target:
                    target_list.append(value)
                target_list = Preprocessor.process_json_node(value, target, target_list)
        else:
            raise NotImplementedError("tree case not implemented: %s" % str(type(node)))
        return target_list
