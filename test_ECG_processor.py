import pytest
from numpy import nan
from numpy import array
import numpy as np


@pytest.mark.parametrize("a, expected", [
    ("a/b/c", 'c'),
    ('a/b/c/', 'c'),
    ('\\a\\b\\c', 'c'),
    ('a/b/../../a/b/c', 'c'),
    ('\\a\\b\\c\\', 'c'),
    ('a/b/../../a/b/c/', 'c'),
    ('a\\b\\c', 'c')
])
def test_path_leaf(a, expected):
    from ECG_processor import path_leaf
    answer = path_leaf(a)
    assert answer == expected


@pytest.mark.parametrize("a, b, expected1, expected2", [
    ([1, 2, nan, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5],
     [1, 2, 4, 5], [0.1, 0.2, 0.4, 0.5]),
    ([2.2, 3.3, 4.4, nan], [1.1, 1.2, 1.3, 1.4],
     [2.2, 3.3, 4.4], [1.1, 1.2, 1.3]),
    ([2.2, 3.3, 4.4, 5.5], [1.1, 1.2, nan, 1.4],
     [2.2, 3.3, 4.4, 5.5], [1.1, 1.2, nan, 1.4]),
    ([nan, 3.3, 4.4, 5.5], [1.1, 1.2, nan, 1.4],
     [3.3, 4.4, 5.5], [1.2, nan, 1.4]),
    ([nan, 3.3, 4.4, nan], [1.1, 1.2, nan, 1.4],
     [3.3, 4.4], [1.2, nan]),
])
def test_if_missing_time(a, b, expected1, expected2):
    from ECG_processor import if_missing_time
    answer1, answer2 = if_missing_time(a, b)
    assert answer1 == expected1
    assert answer2 == expected2


@pytest.mark.parametrize("a, b, expected1, expected2", [
    ([1, 2, nan, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5],
     [1, 2, nan, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5]),
    ([2.2, 3.3, 4.4, 5.5], [1.1, 1.2, nan, 1.4],
     [2.2, 3.3, 5.5], [1.1, 1.2, 1.4]),
    ([nan, 3.3, 4.4, 5.5], [1.1, 1.2, nan, 1.4],
     [nan, 3.3, 5.5], [1.1, 1.2, 1.4]),
    ([nan, 3.3, 4.4, nan], [1.1, 1.2, nan, 1.4],
     [nan, 3.3, nan], [1.1, 1.2, 1.4]),
    ([nan, 3.3, 4.4, nan], [nan, 1.2, nan, 1.4],
     [3.3, nan], [1.2, 1.4]),
])
def test_if_missing_vol(a, b, expected1, expected2):
    from ECG_processor import if_missing_vol
    answer1, answer2 = if_missing_vol(a, b)
    assert answer1 == expected1
    assert answer2 == expected2


@pytest.mark.parametrize("a, b, expected1, expected2", [
    (np.linspace(0, 10, 10), np.sin(np.linspace(0, 10, 10)),
     array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]),
     array([0.222+0.j, 0.18+0.233j, -0.03+0.628j, -2.448+3.508j,
            1.582-0.799j, 1.209+0.j, 1.582+0.799j, -2.448-3.508j,
            -0.03-0.628j, 0.18-0.233j])),
])
def test_fourier_transform(a, b, expected1, expected2):
    from ECG_processor import fourier_transform
    answer1, answer2 = fourier_transform(a, b)
    assert (answer1 == expected1).any()
    new_answer = np.zeros(answer2.shape[0], dtype=complex)
    for i in range(answer2.shape[0]):
        new_answer[i] = round(answer2[i], 3)
    assert (new_answer == expected2).any()


@pytest.mark.parametrize("a, b, c, expected", [
    (array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]),
     np.sin(np.linspace(0, 10, 10)),
     np.fft.fftshift(np.fft.fft(np.sin(np.linspace(0, 10, 10)))),
     array([0.01920249-0.06275656j, 0.03840714+0.0165363j,
            -0.01225143+0.05253656j, -0.06152333-0.04900568j,
            0.08096292-0.02224938j, -0.01920249+0.06275656j,
            -0.03840714-0.0165363j, 0.01225143-0.05253656j,
            0.06152333+0.04900568j, -0.08096292+0.02224938j])),
])
def test_ideal_filter(a, b, c, expected):
    from ECG_processor import ideal_filter
    answer = ideal_filter(a, b, c)
    new_answer = np.zeros(answer.shape[0], dtype=complex)
    for i in range(answer.shape[0]):
        new_answer[i] = round(answer[i], 8)
    assert (new_answer == expected).any()


@pytest.mark.parametrize("a, expected1, expected2, expected3, expected4", [
    (array([1, 2, 1, 3, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3,
            1]),
     array([5, 5]),
     array([0., 0., 0., 0., 0., 1., 0., 1.]),
     [2, 2, 2, 2, 2, 3, 2, 3],
     [3, 3, 3]),
])
def test_find_R_wave(a, expected1, expected2, expected3, expected4):
    from ECG_processor import find_R_wave
    answer1, answer2, answer3, answer4 = find_R_wave(a)
    assert (answer1 == expected1).any()
    assert (answer2 == expected2).any()
    assert (answer3 == expected3)
    assert (answer4 == expected4)


@pytest.mark.parametrize(("a, b, c, d, e, f, r1, r2, r3, r4"), [
    (array([5, 5]),
     array([0., 0., 0., 0., 0., 1., 0., 1.]),
     [2, 2, 2, 2, 2, 3, 2, 3],
     [3, 3, 3],
     np.linspace(0, 10, 23),
     array([1, 2, 1, 3, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3,
            1]),
     10.0, 5, 30,
     array([1.36363636, 1.36363636, 1.36363636, 1.36363636, 1.36363636,
            4.09090909, 4.09090909, 4.09090909, 4.09090909, 4.09090909,
            5.90909091, 5.90909091, 5.90909091, 5.90909091, 5.90909091,
            7.72727273, 7.72727273, 7.72727273, 7.72727273, 7.72727273,
            9.54545455, 9.54545455, 9.54545455, 9.54545455, 9.54545455])),
])
def test_fetch_metrics(a, b, c, d, e, f, r1, r2, r3, r4):
    from ECG_processor import fetch_metrics
    answer1, answer2, answer3, answer4 = fetch_metrics(a, b, c, d, e, f)
    assert (answer1 == r1)
    assert (answer2 == r2)
    assert (answer3 == r3)
    answer4 = np.array(answer4)
    new_answer = np.zeros(answer4.shape[0])
    for i in range(answer4.shape[0]):
        new_answer[i] = round(answer4[i], 8)
    assert (new_answer == r4).any()


@pytest.mark.parametrize("a, b, c, d, e, expected", [
    (10.0, (3, 1), 5, 30,
     [1.36363636, 1.36363636, 1.36363636, 1.36363636, 1.36363636,
      4.09090909, 4.09090909, 4.09090909, 4.09090909, 4.09090909,
      5.90909091, 5.90909091, 5.90909091, 5.90909091, 5.90909091,
      7.72727273, 7.72727273, 7.72727273, 7.72727273, 7.72727273,
      9.54545455, 9.54545455, 9.54545455, 9.54545455, 9.54545455],
     {'duration': 10.0,
      'voltage_extremes': (3, 1),
      'num_beats': 5,
      'mean_hr_bpm': 30,
      'beats': [1.36363636,
                1.36363636,
                1.36363636,
                1.36363636,
                1.36363636,
                4.09090909,
                4.09090909,
                4.09090909,
                4.09090909,
                4.09090909,
                5.90909091,
                5.90909091,
                5.90909091,
                5.90909091,
                5.90909091,
                7.72727273,
                7.72727273,
                7.72727273,
                7.72727273,
                7.72727273,
                9.54545455,
                9.54545455,
                9.54545455,
                9.54545455,
                9.54545455]}),
])
def test_produce_dict(a, b, c, d, e, expected):
    from ECG_processor import produce_dict
    answer = produce_dict(a, b, c, d, e)
    assert (answer == expected)
