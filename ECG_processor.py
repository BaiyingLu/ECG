import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def interface():
    """Take in the data file name

    This function is an interface which can interact with the user. This
    function takes in the file path with file extension.
    Then it calls the function import_data and tsh_info_process

    """
    print("Please input the document path:")
    print("Please include the extension,like: .txt")
    path = input("The path is: ")
    data = take_in_data(path)
if __name__ == "__main__":
    interface()
