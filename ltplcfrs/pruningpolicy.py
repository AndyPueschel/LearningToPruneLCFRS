"""This module contains the pruning policy and its needed subfunction
readoff and a Mode enumeration."""

from enum import Enum


class PruningPolicy(object):
    """The Pruning Policy class represents a collection of pruning decisions
    for nonterminals and their spanning subsentences in a parsing process.

    Attributes
    ----------
    contents : list(tuple(str))
        The list of items of the form ((subsent1), ..., (nonterminal)).
    mode : MODE
        The behavior for the items.

    """

    def __init__(self, contents=None, mode=None):
        """Initialize an instance of a pruning policy.

        Parameters
        ----------
        contents : list(str)
            The lines of contents (items) for the pruning policy.
        mode : MODE
            The behavior for the items collected by the pruning policy.

        """
        if contents:
            if mode:
                self.mode = mode
            else:
                self.mode = Mode.KEEPING
            self.contents = readoff(contents)
        else:
            if mode:
                self.mode = mode
            else:
                self.mode = Mode.PRUNING
            self.contents = []

    def ispruned(self, item):
        """Check whether an item should be pruned.
        Return true if the pruning policy decides to prune the item,
        otherwise return false.

        Parameters
        ----------
        item : tuple(str)
            The item with the form: (subsent1, subsent2, ..., nonterminal).

        Returns
        -------
        bool
            The truth value expressing whether an item will be pruned.

        """
        if self.mode == Mode.PRUNING:
            return item in self.contents
        elif self.mode == Mode.KEEPING:
            return not (item in self.contents)
        else:
            raise ValueError("unknown mode: %s" % self.mode)

    def __str__(self):
        """Return the string representation of the contents.

        Returns
        -------
        str
            The string representation of the contents.

        """
        result = []
        for item in self.contents:
            words = ['(' + w + ')' for w in item]
            line = ''.join(words)
            result.append(line)
        return '\n'.join(result)


def readoff(contents):
    """Convert a list of strings to a list of items for a pruning policy.

    Parameters
    ----------
    contents : list(str)
        The list of items in string format.

    Returns
    -------
    list(tuple(str))
        The list of items.

    """
    result = []
    for line in contents:
        item = []
        word = []
        for char in list(line):
            if char == "(":
                word = []
            elif char == ")":
                item.append(''.join(word))
            else:
                word.append(char)
        result.append(tuple(item))
    return result


class Mode(Enum):
    """The Mode class contains the different pruning modes."""

    PRUNING = 0
    KEEPING = 1


__all__ = ['PruningPolicy', 'readoff', 'Mode']
