"""This module contains the Parser class for parsing a sentence with a given
grammar and pruning policy."""

from itertools import product, dropwhile
from operator import itemgetter

from discodop.tree import escape

from .hypergraph import Hypergraph, Hyperedge, Hypernode
from .pruningpolicy import PruningPolicy
from .features import FeatureItem


class Parser(object):
    """The Parser class contains the function to parse a sentence with
    a given grammar and pruning policy.

    Attributes
    ----------
    grammar : Grammar
        The grammar where the parsing process is based on.

    """

    def __init__(self, grammar):
        """initialize the parser for a given grammar.

        Parameters
        ----------
        grammar : Grammar
            The grammar to use for parsing.

        """
        self.grammar = grammar

    def parse(self, sentence, pp=PruningPolicy()):
        """parses a given sentence with the specified grammar.

        Parameters
        ----------
        sentence : str
            The sentence to parse.
        pp : PruningPolicy
            The pruning policy to use.

        Returns
        -------
        Hypergraph
            The parse forest as hypergraph.

        """
        hg = Hypergraph(sentence)
        slist = list(map(lambda s: escape(s), sentence.split()))
        agenda = []  # 'queue' of hyperedges
        visited = []  # remembers already visited items

        # create leaves in derivation graph.
        trules = self.grammar.get_terminalrules()
        rules = [r for r in trules if r.yf[0] in slist]
        for pos in hg.get_nodes().keys():  # pos is a singleton tuple
            for rule in [r for r in rules if slist[pos[0]] == r.yf[0]]:
                nt = rule.lhs
                p = rule.get_prob()
                node = hg.get_nodes()[pos]
                edge = Hyperedge(nt, p, 1, node)
                visited.append(edge.get_signature())
                agenda.append(edge)

        # create the other items for derivation graph.
        while agenda:
            item = agenda.pop(0)
            hg.add_edge(item)

            # set of (cover, nonterminal) tuples
            tmpnodes = [(c.get_predecessor().get_label(), c.get_nonterminal())
                        for c in hg.get_edges()]
            tmpitem = (item.get_predecessor().get_label(),
                       item.get_nonterminal())
            tmpnodes.append(tmpitem)
            nodes = set(tmpnodes)
            citem = (item.get_nonterminal(),
                     item.get_predecessor().get_label())

            # regarding continuous rules
            for crule in self.grammar.get_continuousrules():

                # create constituents according to the arity of the rule
                constituents = []
                for idx in range(len(crule.rhs)):
                    constituents.append(
                        [c for c, n in nodes if n == crule.rhs[idx]]
                    )

                # iterate through every (cover_1, ..., cover_n) pair
                for pair in list(product(*constituents)):

                    # create feature item
                    fitem = FeatureItem(crule.lhs,
                                        list(zip(crule.rhs, list(pair))),
                                        slist)

                    # validations
                    if not validate_elem(fitem, citem) or\
                       not validate_cover(fitem) or\
                       not validate_continuity(fitem) or\
                       not validate_continuous_run(fitem, crule):
                        continue
                    # check whether the current item should be pruned
                    if pp.ispruned(fitem):
                        continue

                    # add hypernode to hypergraph
                    if not (fitem.cover in hg.get_nodes()):
                        hg.add_node(Hypernode(fitem.cover))
                    node = hg.get_nodes()[fitem.cover]

                    # create signature
                    succs = list(map(lambda x: hg.get_nodes()[x], list(pair)))
                    sign = [(node.get_label(), crule.lhs)]
                    sign_rhs = list(zip(map(Hypernode.get_label, succs),
                                        crule.rhs))
                    sign.extend(sign_rhs)
                    sign = tuple(sign)

                    # create and enqueue the new item if not already visited
                    if sign not in visited:
                        he = Hyperedge(
                                crule.lhs, crule.get_prob(),
                                1, node, list(zip(succs, crule.rhs))
                                )
                        visited.append(he.get_signature())
                        agenda.append(he)

            # regarding discontinuous rules
            for drule in self.grammar.get_discontinuousrules():

                # create constituents according to the arity of the rule
                constituents = []
                for idx in range(len(drule.rhs)):
                    constituents.append(
                        [c for c, n in nodes if n == drule.rhs[idx]]
                    )

                # iterate through every (cover_1, ..., cover_n) pair
                for pair in list(product(*constituents)):
                    cover = [pos for cs in list(pair) for pos in cs]
                    cover.sort()

                    # create feature item
                    fitem = FeatureItem(drule.lhs,
                                        list(zip(drule.rhs, list(pair))),
                                        slist)

                    # validations
                    if not validate_elem(fitem, citem) or\
                       not validate_cover(fitem) or\
                       not validate_discontinuous_run(fitem, drule):
                        continue
                    # check whether the current item should be pruned
                    if pp.ispruned(fitem):
                        continue

                    # add hypernode to hypergraph
                    if not (fitem.cover in hg.get_nodes()):
                        hg.add_node(Hypernode(fitem.cover))
                    node = hg.get_nodes()[fitem.cover]

                    # create signature
                    succs = list(map(lambda x: hg.get_nodes()[x], list(pair)))
                    sign = [(node.get_label(), crule.lhs)]
                    sign_rhs = list(zip(map(Hypernode.get_label, succs),
                                        crule.rhs))
                    sign.extend(sign_rhs)
                    sign = tuple(sign)

                    # create and enqueue the new item if not already visited
                    if sign not in visited:
                        he = Hyperedge(
                                drule.lhs, drule.get_prob(),
                                1, node, list(zip(succs, drule.rhs))
                                )
                        visited.append(he.get_signature())
                        agenda.append(he)
        return hg


def validate_cover(fitem):
    if not fitem.cover:
        return False
    if len(fitem.cover) != len(set(fitem.cover)):
        return False
    else:
        return True


def validate_continuity(fitem):
    if max(fitem.cover) - min(fitem.cover) + 1 != len(fitem.cover):
        return False
    else:
        return True


def validate_continuous_run(fitem, crule):
    interval = [(idx, n)
                for n, t in enumerate([c for _n, c in fitem.rhs])
                for idx in t]
    interval.sort(key=itemgetter(0))
    run = []
    for child in crule.yf[0]:
        if interval:
            run.append(interval[0][1])
        interval =\
            list(dropwhile(lambda x: x[1] == child, interval))
    # if yf is valid, the run equals the yield function
    if tuple(run) != crule.yf[0]:
        return False
    else:
        return True


def validate_discontinuous_run(fitem, drule):
    tmpcover = [(idx, n)
                for n, t in enumerate([c for _n, c in fitem.rhs])
                for idx in t]
    tmpcover.sort(key=itemgetter(0))
    # create interval of subintervals
    tmpprev = fitem.cover[0] - 1
    tmpinterval = []
    intervals = []
    for pos, c in tmpcover:
        if pos == tmpprev + 1:
            tmpinterval.append((pos, c))
        else:
            intervals.append(tmpinterval)
            tmpinterval = [(pos, c)]
        tmpprev = pos
    intervals.append(tmpinterval)
    # number of intervals must match the intervals in yf
    if len(intervals) != len(drule.yf):
        return False
    # validate the yield function of the rule
    run = []
    for csid in range(len(drule.yf)):
        subrun = []
        for child in drule.yf[csid]:
            if tmpcover:
                subrun.append(tmpcover[0][1])
            tmpcover = list(dropwhile(lambda x: x[1] == child, tmpcover))
        run.append(tuple(subrun))
    # if yf is valid, the run equals the yield function
    if tuple(run) != drule.yf:
        return False
    else:
        return True


def validate_elem(fitem, citem):
    if citem not in fitem.rhs:
        return False
    else:
        return True


__all__ = ['Parser']
