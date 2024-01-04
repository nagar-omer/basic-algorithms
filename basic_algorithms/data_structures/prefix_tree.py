from dataclasses import dataclass, field

@dataclass(order=True)
class TreeNode:
    """
    Helper class for max sequence up to n items.
    """
    letter: chr = field(default=None, compare=True)
    is_word: bool = field(default=False, compare=False)
    children: dict = field(default_factory=dict, compare=False)


class PrefixTree:
    def __init__(self):
        self._root = TreeNode()

    def insert(self, word):
        """
        Insert word into the prefix tree
        :param word: word to insert
        :return: None
        """

        # start at the root
        node = self._root
        # for each letter in the word
        for letter in word:
            # create a new node if the letter is not in the children
            if letter not in node.children:
                node.children[letter] = TreeNode(letter)
            node = node.children[letter]

        # mark the last node as a word
        node.is_word = True

    def contains(self, word):
        """
        Search for a word in the prefix tree
        :param word: word to search
        :return: True if word is found, False otherwise
        """
        # start at the root
        node = self._root
        # for each letter in the word
        for letter in word:
            # return False if the next letter is not in the children
            if letter not in node.children:
                return False
            node = node.children[letter]
        # if final node is a word, return True
        return node.is_word


if __name__ == '__main__':
    prefix_tree = PrefixTree()
    prefix_tree.insert("abc")
    prefix_tree.insert("abcd")
    prefix_tree.insert("abef")
    assert not prefix_tree.contains("efg")
    assert prefix_tree.contains("abc")
    assert not prefix_tree.contains("ab")
    assert prefix_tree.contains("abcd")
    assert prefix_tree.contains("abef")