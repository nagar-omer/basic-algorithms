import re
# useful links:
# https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285
# https://regex101.com/r/cO8lqs/24
# https://regexr.com


def starts_with(prefix):
    return f'^{prefix}'


def ends_with(suffix):
    return f'{suffix}$'


def exact_match(pattern):
    return f'^{pattern}$'


def lang1():
    # matches the following lang a followed by 0 or more b's
    return r'ab*'


def lang2():
    # matches the following lang a followed by 1 or more b's
    return r'ab+'


def lang3():
    # matches the following lang a followed by 0 or 1 b's
    return r'ab?'


def lang4():
    # matches the following lang a followed by exactly 3 b's and then zero or more c
    return r'ab{3}c*'


def lang5():
    # matches the following lang a followed by exactly [2-4] b's and then zero or more c
    return r'ab{2,4}c*'


def lang6():
    # matches the following lang a followed 1+ times .tar.gz
    return r'a(.tar.gz)+'


def lang7():
    # matches  a - zero or more a/b's - a
    # or       b - zero or more a/b's - b
    return '(a[ab]*a)|(b[ab]*b)'


def lang8():
    # a followed by 3-5 any character followed by b
    return r'a.{3,5}b'


def lang9():
    # first & last name separated by a spaces followed by a :  up to 4 spaces
    # followed by a 10 digit phone number (w/wo -)
    # first & last name up to 20 characters each
    # e.g. John Smith: 054-1234567
    first_last_name = r'\w{1,20}\s+\w{1,20}'
    phone_number = r'\d{3}-?\d{7}'
    return f'{first_last_name}:\s{{1,4}}{phone_number}'


def lang10():
    # digit not digit sequence
    return r'(\D(\d\D)*\d?)|(\d(\D\d)*\D?)'


def lang11():
    # a-f followed by not a-f 2 letters followed by 0-3 numbers
    return r'[a-f][^a-f\d]{2}\d{0,3}'


# <.*?> matches any character between < and >, but as few characters as possible
# <.*> matches any character between < and >, but as many characters as possible
# <[^<>]*> matches any character between < and >, but not < or >
# /Babc/B matches abc only if it is not preceded or followed by another letter


def match(text, pattern):
    """
    Check if the text matches the pattern
    :param text: text to match
    :param pattern: pattern to match
    :return: True if the text matches the pattern, False otherwise
    """
    # match the entire text
    pos = re.match(pattern, text)
    # if no match, return False
    if pos is None:
        return False

    # if match, check if the match is the entire text
    start, end = pos.span()
    if start == 0 and end == len(text):
        return True
    return False


if __name__ == '__main__':
    text = 'abbbccccc'
    print(f'text: {text}, pattern: {lang4()}', match(text, lang4()))