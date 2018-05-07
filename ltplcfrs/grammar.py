"""This module contains the Grammar class for initializing a grammar
and further functions to operate on rules."""

from discodop.grammar import treebankgrammar


class Grammar(object):
    """The Grammar class contains the rules for a LCFRS.
    It splits the rules into three clusters according to their continuouity:
    pos rules, rules with a continuous cover on the lhs nonterminal and
    rules with a discontinuous cover on the lhs nonterminal.

    Attributes
    ----------
    terminal_rules : list(((tuple(str), (str,)), double))
        The list of pos rules.
    continuous_rules : list(((tuple(str), (tuple(int),)), double))
        The list of continuous rules.
    discontinuous_rules : list(((tuple(str), tuple(tuple(int))),double))
        The list of discontinuous_rules.

    """

    def __init__(self, trees, sentences):
        """Induce an instance of a grammar by using tree-sentence pairs.

        Parameters
        ----------
        trees: list(Tree)
            The list of gold trees.
        sentences: list(str)
            The list of sentences.

        The rules in the grammar have the following form:
        rule pos:   (((lhs, 'Epsilon' ), (word,           )), probability)
        rule other: (((lhs, rhs_1, ...), ((1, 2, 0), (0, ))), probability)

        """
        # creating basic rules
        rules = treebankgrammar(trees, sentences)

        # calculating relative frequencies
        trules = []
        crules = []
        drules = []
        for rule in rules:
            r = Rule(rule)
            if r.rhs[0] == "Epsilon":
                trules.append(r)
            elif len(r.yf) == 1:
                crules.append(r)
            else:
                drules.append(r)

        # assign values
        self.discontinuous_rules = drules
        self.continuous_rules = crules
        self.terminal_rules = trules

    def get_terminalrules(self):
        """Return the list of pos rules.

        Returns
        -------
        list((tuple(str), tuple(str)), double)
            The list of pos rules.

        """
        return self.terminal_rules

    def get_continuousrules(self):
        """Return the list of rules with continuous cover on the lhs
        nonterminal, except the pos rules.

        Returns
        -------
        list((tuple(str), tuple(tuple(int))), double)
            The list of rules with continuous cover on the lhs nonterminal.

        """
        return self.continuous_rules

    def get_discontinuousrules(self):
        """Return the list of rules with discontinuous cover on the lhs
        nonterminal.

        Returns
        -------
        list((tuple(str), tuple(tuple(int))), double)
            The list of rules with discontinuous cover on the lhs nonterminal.

        """
        return self.discontinuous_rules

    def get_rules(self):
        """Return the all rules of the grammar.

        Returns
        -------
        list((tuple(str), tuple(str | tuple(int))), double)
            The list of rules in the grammar.

        """
        return self.terminal_rules \
            .extend(self.continuous_rules) \
            .extend(self.discontinuous_rules)


class Rule(object):
    """The Rule class contains / restructures the information given by
    a rule according to the discodop library.

    Attributes
    ----------
    lhs : str
        The left hand side of the rule.
    rhs : tuple(str)
        The right hand side of the rule.
    yf : tuple(str) or tuple(tuple(int))
        The yield function of the rule.
    prob : (int, int)
        The probability (weight) of the rule.

    """

    def __init__(self, discodop_rule):
        """Initialize an instance of a rule.

        Parameters
        ----------
        discodop_rule : list((tuple(str), tuple(str | tuple(int))), double)
            The rule according to the discodop library.

        """
        if not isinstance(discodop_rule, tuple):
            self.lhs = "None"
            self.rhs = []
            self.prob = (0, 1)
            self.yf = tuple(())
        else:
            # p has the form (numerator, denominator)
            (nts, yf), p = discodop_rule
            self.lhs = nts[0]
            self.rhs = list(nts[1:])
            self.prob = p
            self.yf = yf

    def get_prob(self):
        """Return the probability.

        Returns
        -------
        double
            The probability (weight) of the rule.

        """
        return float(self.prob[0] / self.prob[1])

    def __str__(self):
        """Return the string representation.

        Returns
        -------
        str
            The string representation.

        """
        return '%s -> %s # %s # %s' %\
            (str(self.lhs),
             ' '.join(self.rhs),
             '-'.join([str(i) for c in self.yf for i in c]),
             ' / '.join([str(p) for p in self.prob]))

    def __repr__(self):
        """Return the string representation for 'repr'.

        Returns
        -------
        str
            The string representation for 'repr'.

        """
        return '%s -> %s' % (self.lhs, ' '.join(self.rhs))


__all__ = ['Grammar', 'Rule']
