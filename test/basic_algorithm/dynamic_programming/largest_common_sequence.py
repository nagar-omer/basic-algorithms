from basic_algorithms.dynamic_programming.largest_common_sequence import largest_common_sequence


def test_empty():
    assert largest_common_sequence('', '') == '', 'empty string'


def test_single_char():
    assert largest_common_sequence('a', 'a') == 'a', 'single char'


def test_complete_match():
    assert largest_common_sequence('abababc', 'abababc') == 'abababc', 'complete match'


def test_partial_match():
    assert largest_common_sequence('ccABA', 'dABAdd') == 'ABA', 'partial match'


def test_no_match():
    assert largest_common_sequence('abababc', 'dssdsd') == '', 'no match'


def test_two_options():
    assert largest_common_sequence('ABABcc', 'ddABmmABABABmm') == 'ABAB', 'two options'
