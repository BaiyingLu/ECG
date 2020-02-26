import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import ntpath
from numpy import nan


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def if_missing_time(time, voltage):
    time_na = []
    count = 0
    for i in range(len(time)):
        if math.isnan(time[i]) is True:
            time_na.append(i)
    for j in time_na:
        print(time_na)
        time.pop(j - count)
        voltage.pop(j - count)
        count = count + 1
    return time, voltage


def take_in_data(path):
    ECG = pd.read_csv(path)
    ECG.columns = ["time", "voltage"]
    ECG.time = pd.to_numeric(ECG.time, errors='coerce')
    ECG.voltage = pd.to_numeric(ECG.voltage, errors='coerce')
    time = list(ECG['time'])
    voltage = list(ECG['voltage'])
    for i in range(len(time)):
        time[i] = float(time[i])
    for i in range(len(voltage)):
        voltage[i] = float(voltage[i])
    if len(time_na) != 0:
        logging.error("There is missing data in time list")
    return time, voltage


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
