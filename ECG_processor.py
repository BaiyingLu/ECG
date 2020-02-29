import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import ntpath
from numpy import nan
from warnings import warn
from scipy.signal import find_peaks
import json


def path_leaf(path):
    """Extract the file name from the file path

    File path is given to the function as a string and
    this function extracts the file name and returns it.

    Args:
        path (string): the inputted file path

    Returns:
        string: the file name
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def if_missing_time(time, voltage):
    """Detect and remove missing data in time column

    The whole data with time column and voltage column is
    given. This function detects the missing data in time
    column and pop the nan out and the corresponding voltage.
    The function also flags in logging file if missing data
    is in the time column.

    Args:
        time (array): the inputted time data
        voltage (array): the inputted voltage data

    Returns:
        array: the time array without missing data
        array: the voltage array
    """
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
    """Detect and remove missing data in voltage column

    The whole data with time column and voltage column is
    given. This function detects the missing data in voltage
    column and pop the nan out and the corresponding time.
    The function also flags in logging file if missing data
    is in the voltage column.

    Args:
        time (array): the inputted time data
        voltage (array): the inputted voltage data

    Returns:
        array: the time array
        array: the voltage array without missing data
    """
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
    """Take in the data in two arrays

    The path of the data is given. The function will take in this data
    and transfer it into two arrays "time" and "voltage".

    Args:
        path (string): the inputted file path

    Returns:
        array: the time array
        array: the voltage array without missing data
    """
    ECG = pd.read_csv(path)
    logging.info("Start a new ECG trace")
    ECG.columns = ["time", "voltage"]
    ECG.time = pd.to_numeric(ECG.time, errors='coerce')
    ECG.voltage = pd.to_numeric(ECG.voltage, errors='coerce')
    time = list(ECG['time'])
    voltage = list(ECG['voltage'])
    for i in range(len(time)):
        time[i] = float(time[i])
    for i in range(len(voltage)):
        voltage[i] = float(voltage[i])
    return time, voltage


def extreme_detection(voltage):
    """Find the maximum and minimum

    The function finds the maximum and minimum from the voltage
    and output them as a tuple.

    Args:
        voltage (array): the inputted file path

    Returns:
        tuple: the voltage extremes with maximum and minimum
    """
    voltage_extremes = (max(voltage), min(voltage))
    if voltage_extremes[0] > 300 or voltage_extremes[1] < -300:
        logging.warning('The voltages exceeded the normal range')
    return voltage_extremes


def fourier_transform(time, voltage):
    """Do Fourier transform to the original signal

    This function calculates the sample frequency and sets
    the frequency index. Then Fast Fourier Transformation will be
    done in this function. The frequency index and the frequency
    spectrum will be returned by this function.

    Args:
        time (array): the inputted time data without missing
        voltage (array): the inputted voltage data without missing

    Returns:
        array: the frequency index
        array: the frequency spectrum of the signal
    """
    t0 = time[1] - time[0]
    f_sample = 1/t0
    f_index = np.linspace(-f_sample, f_sample, len(voltage))
    freq_ECG = np.fft.fftshift(np.fft.fft(voltage))
    return f_index, freq_ECG


def ideal_filter(f_index, voltage, freq_ECG):
    """Pass the signal through an ideal filter

    This function takes in the fourier transformed data
    and passes it through a inner ideal band-pass filter, which has the
    high cutoff frequency 45Hz and low cutoff frequency 0.7Hz.
    The filtered signal will be recovered by inverse fast fourier
    tansform. The recovered signal will be returned.

    Args:
        f_index (array): the frequency index
        voltage (array): the inputted voltage data without missing
        freq_ECG (array): the frequency spectrum of the signal

    Returns:
        array: the filtered recovered signal
    """
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
    """Find the R peaks in the sequence

    This function will find the R peaks from ECG signal.
    This fucntion uses find_peaks function in scipy.signal package. Then
    it will extract the three largest value and do normalization to the
    rest of peaks. Then the function extracts the peaks again and attach
    the three largest value to the list of peaks.
    The function returns the result index after second findpeaks, the
    normalized voltage, the result of second findpeaks with the attached
    peaks.

    Args:
        recovered_time (array): the filtered recovered signal

    Returns:
        array: the result index after second findpeaks
        array: the normalized voltage
        list: the result of second findpeaks
        list: the three largest value before normalization
    """
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
    """Find the duration, num_beats, mean_hr_bpm, beats_time

    This fucntion receives the result index after second findpeaks,
    the normalized voltage, the result of second findpeaks,
    the three largest value before normalization, the original
    time sequence without missing data, the recovered signal from
    filter. The function calculates the duration: time duration of
    the ECG strip, the num_beats: number of detected beats in the
    strip, as a numeric variable type, the mean_hr_bpm:
    estimated average heart rate over the length of the strip,
    and the beats: the list of times when a beat occurred.

    Args:
        new_peaks (array): the result index after second findpeaks
        normalized_voltage (array): the normalized voltage
        wrapped_voltage (list): the result of second findpeaks
        value (list): the three largest value before normalization
        recovered_time (array): the filtered recovered signal
        time (array): the inputted time data without missing

    Returns:
        float: time duration of the ECG strip
        int: number of detected beats in the strip, as a numeric variable type
        int: estimated average heart rate over the length of the strip
        list: list of times when a beat occurred
    """
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


def produce_dict(duration, voltage_extremes, num_beats,
                 mean_hr_bpm, beats_time):
    """Put the metrics into dictionary

    This function put the duration: time duration of
    the ECG strip, voltage_extremes: tuple in the form
    (min, max) where min and max are the minimum and maximum
    lead voltages found in the data filethe num_beats:
    number of detected beats in the strip, as a numeric variable
    type, the mean_hr_bpm: estimated average heart rate over the
    length of the strip, and the beats: the list of times when a
    beat occurred.

    Args:
        duration (float): time duration of the ECG strip
        voltage_extremes (tuple): the voltage extremes with maximum and
        minimum
        num_beats (int): number of detected beats in the strip,
        as a numeric variable type
        mean_hr_bpm (int): estimated average heart rate over the length of
        the strip
        beats_time (list): list of times when a beat occurred

    Returns:
        dictionary: the dictionary with different metrics of an ECG signal
    """
    logging.info("Assign the ECG trace metrics into dictionary")
    patient_dict = {"duration": duration,
                    "voltage_extremes": voltage_extremes,
                    "num_beats": num_beats,
                    "mean_hr_bpm": mean_hr_bpm,
                    "beats": beats_time}
    return patient_dict


def output_file(patient_dict, file_name):
    """Output the signal information in dictionary into .json file

    This function takes in the dictionary with different metrics
    and create a .json file named by the data file name.
    Then close the created file.

    Args:
        patient_dict (dictionary): the dictionary stores the signal information
        file_name (string): the file name

    """
    filename = file_name + '.json'
    out_file = open(filename, "w")
    json.dump(patient_dict, out_file)
    out_file.close()


def interface():
    """Take in the data file name
    This function is an interface which can interact with the user. This
    function takes in the file path with file extension.
    Then it calls the function extreme_detection to collect the maximum
    and the minimum.
    Then it calls the functions if_missing_time and if_missing_vol to
    remove the missing data.
    Then it calls the function fourier_transform to do fast fourier
    transform to the data without missing data.
    Then it calls the function ideal_filter to pass the signal through
    an ideal filter to remove the baseline of the signal.
    Then it calls the function find_R_wave to find the R peaks.
    Then it calls the function fetch_metrics to calculate the metrics.
    Then it calls the function produce_dict to output the metrics into
    dictionary.
    Then it calls the fucntion output_file to output .json file.

    """
    print("Please input the document path:")
    print("Please include the extension,like: .csv")
    path = input("The path is: ")
    file_name = path_leaf(path)
    logging.basicConfig(filename=file_name + '.log',
                        level=logging.INFO,
                        filemode='w')
    time, voltage = take_in_data(path)
    time, voltage = if_missing_time(time, voltage)
    time, voltage = if_missing_vol(time, voltage)
    voltage_extremes = extreme_detection(voltage)
    f_index, freq_ECG = fourier_transform(time, voltage)
    recovered_time = ideal_filter(f_index, voltage, freq_ECG)
    (new_peaks, normalized_voltage,
     wrapped_voltage, value) = find_R_wave(recovered_time)
    (duration, num_beats, mean_hr_bpm,
     beats_time) = fetch_metrics(new_peaks, normalized_voltage,
                                 wrapped_voltage, value,
                                 time, recovered_time)
    patient_dict = produce_dict(duration, voltage_extremes, num_beats,
                                mean_hr_bpm, beats_time)
    output_file(patient_dict, file_name)
if __name__ == "__main__":
    interface()
