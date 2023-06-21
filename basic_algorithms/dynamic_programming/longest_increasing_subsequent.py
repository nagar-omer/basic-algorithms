from dataclasses import dataclass, field


@dataclass(order=True)
class NumItem:
    """
    Helper class for max sequence up to n items.
    """
    num: int
    last_len: int = field(compare=False)
    max_seq_len: int = field(compare=False)
    max_end_pos: int = field(compare=False)


def longest_increasing_subsequence(num_list):
    """
    Dynamic programming solution for longest increasing subsequence (LIS) problem.
    LIS[0...n] =
        1. n is last element in the increasing subsequence:
            LIS[n] = LIS[n-1].last + 1
        2. n not last in increasing subsequence:
            LIS[n] = LIS[n-1].max

    :param num_list: list of numbers
    return: longest increasing subsequence
    """
    if len(num_list) <= 1:
        return []

    progress = [NumItem(num_list[0], last_len=0, max_seq_len=0, max_end_pos=0)] + \
               [NumItem(n, last_len=-1, max_seq_len=-1, max_end_pos=-1) for n in num_list[1:]]
    for i in range(1, len(progress)):
        prev_num, num = progress[i-1], progress[i]
        # not increasing pair -> not last in increasing subsequence
        if prev_num >= num:
            num.last_len = 0
            num.max_seq_len = prev_num.max_seq_len
            num.max_end_pos = prev_num.max_end_pos
        # increasing pair -> possibly last in increasing subsequence
        else:
            # min sequence length is 2
            num.last_len = max(2, prev_num.last_len + 1)
            # sequence ending at i is largest so far -> not last in increasing subsequence
            if prev_num.max_seq_len > num.last_len:
                num.max_seq_len = prev_num.max_seq_len
                num.max_end_pos = prev_num.max_end_pos
            # last element in the increasing subsequence
            else:
                num.max_seq_len = num.last_len
                num.max_end_pos = i

    return num_list[progress[-1].max_end_pos - progress[-1].last_len + 1: progress[-1].max_end_pos + 1]
