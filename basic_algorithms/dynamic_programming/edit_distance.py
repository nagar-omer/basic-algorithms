import numpy as np


def edit_distance(s1, s2, w_delete=1, w_insert=1, w_replace=1):
    """
    Given two strings s1 ans s2, edit distance is the minimum number of edits (weighted) needed to transform s1 into s2.
    possible operations are:
        - inserting a single symbol
        - deletion of a single symbol
        - substitution of a single symbol

    D[i, j] = edit distance between s1[0..i-1] and s2[0..j] +
        1. i==j:    0
        i  i!=l: MIN
            a. del i:  w_del
            b. insert j:  w_ins
            c. replace i with j:  w_rep
    """

    # container to calculate edit distance
    edit_dist_mat = np.zeros((len(s1)+1, len(s2)+1))

    # initialize edit distance matrix -> insert weight
    for j in range(1, len(s2)+1):
        edit_dist_mat[0, j] = edit_dist_mat[0, j-1] + w_insert

    # initialize edit distance matrix -> delete weight
    for i in range(1, len(s1)+1):
        edit_dist_mat[i, 0] = edit_dist_mat[i-1, 0] + w_delete

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            # same symbol -> no edit
            if s1[i-1] == s2[j-1]:
                edit_dist_mat[i, j] = edit_dist_mat[i-1, j-1]
            else:
                # apply minimal weight edit
                edit_dist_mat[i, j] = min(
                    edit_dist_mat[i-1, j-1] + w_replace,
                    edit_dist_mat[i-1, j] + w_delete,
                    edit_dist_mat[i, j-1] + w_insert
                )

    return edit_dist_mat[len(s1), len(s2)]
