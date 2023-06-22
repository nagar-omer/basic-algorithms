from basic_algorithms.dynamic_programming.edit_distance import edit_distance


def test_edit_distance():
    assert edit_distance("", "") == 0
    assert edit_distance("", "a") == 1
    assert edit_distance("a", "") == 1
    assert edit_distance("a", "a") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("abc", "acd") == 2
    assert edit_distance("abc", "bcd") == 2
    assert edit_distance("abc", "acd") == 2


def test_complex_string_edit_distance():
    assert edit_distance("lewenstein", "levenshtein") == 2
    assert edit_distance("edit_dddictanse", "edit_dictanse") == 2
    assert edit_distance("edit_dddictanse", "edit_dictanse", w_insert=1, w_delete=2, w_replace=9) == 4
    assert edit_distance("edit_dddictanse", "levin_distance", w_insert=6, w_delete=2, w_replace=9) == 42
