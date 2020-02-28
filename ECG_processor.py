import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import ntpath
from numpy import nan
from warnings import warn
from scipy.signal import find_peaks


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
        time.pop(j - count)
        voltage.pop(j - count)
        count = count + 1
    if len(time_na) != 0:
        logging.error("There is missing data in time list")
    return time, voltage


def if_missing_vol(time, voltage):
    vol_na = []
    count = 0
    for i in range(len(voltage)):
        if math.isnan(voltage[i]) is True:
            vol_na.append(i)
    for j in vol_na:
        time.pop(j - count)
        voltage.pop(j - count)
        count = count + 1
    if len(vol_na) != 0:
        logging.error("There is missing data in voltage list")
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


def extreme_detection(voltage):
    voltage_extremes = (max(voltage), min(voltage))
    if voltage_extremes[0] > 300 | voltage_extremes[1] < -300:
        logging.warning('The voltages exceeded the normal range')
    return voltage_extremes


def fourier_transform(time, voltage):
    t0 = time[1] - time[0]
    f_sample = 1/t0
    f_index = np.linspace(-f_sample, f_sample, len(voltage))
    freq_ECG = np.fft.fftshift(np.fft.fft(voltage))
    return f_index, freq_ECG


def ideal_filter(f_index, voltage, freq_ECG):
    hz_minus50 = np.where(f_index >= -45)
    hz_minus50 = hz_minus50[0][0]
    hz_50 = np.where(f_index <= 45)
    hz_50 = hz_50[0][-1]
    hz_minus05 = np.where(f_index >= -0.7)
    hz_minus05 = hz_minus05[0][0]
    hz_05 = np.where(f_index <= 0.7)
    hz_05 = hz_05[0][-1]
    ideal_filter = np.zeros(len(voltage))
    ideal_filter[hz_minus50:hz_minus05] = 1
    ideal_filter[hz_05:hz_50] = 1
    after_filter = freq_ECG * ideal_filter
    recovered_time = np.fft.ifft(np.fft.ifftshift(after_filter))
    return recovered_time


def find_R_wave(recovered_time):
    peaks, _ = find_peaks(recovered_time)
    recovered_time = np.array(recovered_time)
    wrapped_voltage = recovered_time[peaks]
    wrapped_voltage = list(wrapped_voltage)
    value = []
    for i in range(3):
        value.append(max(wrapped_voltage))
        wrapped_voltage.remove(max(wrapped_voltage))
    min_v = min(wrapped_voltage)
    max_v = max(wrapped_voltage)
    normalized_voltage = ((wrapped_voltage-min_v)/(max_v-min_v))
    new_peaks, _ = find_peaks(normalized_voltage, height=0.7)
    if np.real(normalized_voltage[0]) > 0.7:
        new_peaks = np.insert(new_peaks, 0, 0)
    if np.real(normalized_voltage[-1]) > 0.7:
        last_i = np.where(normalized_voltage == normalized_voltage[-1])[0][0]
        new_peaks = np.append(new_peaks, last_i)
    return new_peaks, normalized_voltage, wrapped_voltage, value


def fetch_metrics(new_peaks, normalized_voltage,
                  wrapped_voltage, value,
                  time, recovered_time):
    num_beats = normalized_voltage[new_peaks].shape
    num_beats = num_beats[0] + 3
    duration = time[-1]
    mean_hr_bpm = (num_beats/duration) * 60
    mean_hr_bpm = round(mean_hr_bpm)
    wrapped_voltage = np.array(wrapped_voltage)
    original_voltage = wrapped_voltage[new_peaks]
    original_voltage = list(original_voltage)
    new_value = original_voltage
    value = value + new_value
    beats = []
    for j in range(0, len(value)):
        for i in range(0, len(recovered_time)):
            if recovered_time[i] == value[j]:
                beats.append(i)
    beats.sort()
    beats_time = []
    for i in beats:
        beats_time.append(time[i])
    return duration, num_beats, mean_hr_bpm, beats_time


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
