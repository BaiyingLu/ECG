import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def interface():
    """Take in the data file name
    This function is an interface which can interact with the user. This
    function takes in the file path with file extension.
    Then it calls the function import_data and tsh_info_process
    """
    print("Please input the document path:")
    print("Please include the extension,like: .txt")
    path = input("The path is: ")
    file_name = path_leaf(path)
    logging.basicConfig(filename=file_name + '.log',
                        level=logging.INFO,
                        filemode='w')
    data = take_in_data(path)
if __name__ == "__main__":
    interface()
