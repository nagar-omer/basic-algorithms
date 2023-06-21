import numpy as np


def reconstruct_lcm(s1, s2, lcm_track_matrix, lcm_length):
    """
    Generate the largest common sequence given two strings & and LCM source matrix
    :param s1: first string
    :param s2: second string
    :param lcm_track_matrix: source matrix (S[i, j] -> argmaxLCM([i-1, j-1], [i-1, j], [i, j-1]))
    :param lcm_length: length of LCM
    :return: the largest common sequence
    """
    # if there is no LCM, return empty string
    if lcm_length == 0:
        return ''

    # find the end of LCM - end index in s1 and s2
    x_loc, y_loc = len(s1) - 1, len(s2) - 1
    while x_loc != -1 or y_loc != -1:
        source_x, source_y = lcm_track_matrix[x_loc, y_loc]
        # letter was added -> this is part of LCM -> reached end of LCM
        if source_x + 1 == x_loc and source_y + 1 == y_loc:
            break
        x_loc, y_loc = source_x, source_y

    # reconstruct the LCM using end point and length
    return s1[x_loc - lcm_length + 1: x_loc + 1]


def largest_common_sequence(s1, s2):
    """
    Implementation of dynamic programming algorithm to find the largest common sequence given two strings.
    :param s1: first string
    :param s2: second string
    :return: the largest common sequence
    """

    # nothing to match -> return empty string
    if len(s1) == 0 or len(s2) == 0:
        return ''

    # Initialization create lcm matrix
    # LCM[i][j] = The largest common divisor of s1[0..i] and s2[0..j]
    # LCM[0][j] =
    #   - if s1[i] == s2[j] = max(LCM[i-1][j-1] + 1)
    #   - if s1[i] != s2[j] = max(LCM[i-1][j-1], LCM[i][j-1], LCM[i-1][j])
    lcm_matrix = np.zeros((len(s1), len(s2))).astype(np.uint8)
    lcm_track_matrix = np.ones((len(s1), len(s2), 2)).astype(np.int8) * -1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            lcm_matrix[i][0] = 1
            lcm_track_matrix[i, 0] = [i-1, -1]

    for j in range(len(s2)):
        if s1[0] == s2[j]:
            lcm_matrix[0][j] = 1
            lcm_track_matrix[0, j] = [-1, j-1]

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            # matching letter -> add 1 to lcm of s1[0..i-1] and s2[0..j-1]
            if s1[i] == s2[j]:
                if lcm_track_matrix[i-1, j-1, 0] == i-2 and lcm_track_matrix[i-1, j-1, 1] == j-2:
                    lcm_matrix[i][j] = lcm_matrix[i-1][j-1] + 1
                else:
                    lcm_matrix[i][j] = 1
                lcm_track_matrix[i][j] = [i-1, j-1]
            # no matching letter -> take max of lcm of s1[0..i-1] | s2[0..j-1]
            elif s1[i] != s2[j]:
                # propagate progress to one direction in order to reconstruct a single sequence
                if lcm_matrix[i-1, j] >= lcm_matrix[i, j-1]:
                    lcm_matrix[i][j] = lcm_matrix[i-1][j]
                    lcm_track_matrix[i][j] = [i-1, j]
                else:
                    lcm_matrix[i][j] = lcm_matrix[i, j-1]
                    lcm_track_matrix[i][j] = [i, j-1]

    lcm = reconstruct_lcm(s1, s2, lcm_track_matrix, lcm_matrix[-1, -1])
    return lcm

