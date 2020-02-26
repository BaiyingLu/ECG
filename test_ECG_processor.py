import pytest
from numpy import nan


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
])
def test_if_missing_time(a, b, expected1, expected2):
    from ECG_processor import if_missing_time
    answer1, answer2 = if_missing_time(a, b)
    assert answer1 == expected1
    assert answer2 == expected2
