"""This module contains the Parser class for parsing a sentence with a given
grammar and pruning policy."""

from itertools import product, dropwhile
from operator import itemgetter
from sys import stdout

from discodop.tree import escape

from .hypergraph import Hypergraph, Hyperedge, Hypernode
from .pruningpolicy import PruningPolicy


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

        # create leaves in derivation graph.
        trules = self.grammar.get_terminalrules()
        rules = [r for r in trules if r.yf[0] in slist]
        for pos in hg.get_nodes().keys():  # pos is a singleton tuple
            for rule in [r for r in rules if slist[pos[0]] == r.yf[0]]:
                nt = rule.lhs
                p = rule.get_prob()
                node = hg.get_nodes()[pos]
                edge = Hyperedge(nt, p, 1, node)
                agenda.append(edge)

        # create the other items for derivation graph.
        while agenda:
            item = agenda.pop(0)
            print("item: %s" % str(item.get_signature()))
            stdout.flush()
            # set of (cover, nonterminal) tuples
            tmpnodes = [(c.get_predecessor().get_label(), c.get_nonterminal())
                        for c in hg.get_edges()]
            tmpitem = (item.get_predecessor().get_label(),
                       item.get_nonterminal())
            tmpnodes.append(tmpitem)
            nodes = set(tmpnodes)

            # regarding continuous rules
            for crule in self.grammar.get_continuousrules():
                # create constituents according to the arity of the rule
                constituents = []
                for idx in range(len(crule.rhs)):
                    constituents.append(
                        [c for c, n in nodes if n == crule.rhs[idx]]
                    )
                pairs = list(product(*constituents))
                for pair in pairs:
                    # print("\npairlist: %s" % str([str(i) for i in pair]))
                    cover = [pos for cs in pair for pos in cs]
                    cover.sort()
                    # cover must not be empty
                    if not cover:
                        continue
                    # positions should be disjoint
                    if len(cover) != len(set(cover)):
                        continue
                    # cover should be continuous
                    if max(cover) - min(cover) + 1 != len(cover):
                        continue
                    # check whether the current item should be pruned
                    subsent = ' '.join([slist[i] for i in cover])
                    if pp.ispruned((subsent, crule.lhs)):
                        continue
                    # validate the yield function of the rule
                    interval = [(idx, n)
                                for n, t in enumerate(list(pair))
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
                        continue

                    # add hypernode to hypergraph
                    label = tuple(cover)
                    if not (label in hg.get_nodes()):
                        hg.add_node(Hypernode(label))
                    node = hg.get_nodes()[label]
                    # create new hyperedge
                    succs = list(map(lambda x: hg.get_nodes()[x], list(pair)))
                    he = Hyperedge(
                            crule.lhs, crule.get_prob(),
                            1, node, zip(succs, crule.rhs)
                            )
                    # enqueue the new item if not already visited
                    joinedlists = list(hg.get_edges())
                    joinedlists.extend(list(agenda))
                    if len(list(filter(
                            lambda x: x.get_signature() == he.get_signature(),
                            joinedlists))) > 1:
                        continue
                    agenda.append(he)

            # regarding discontinuous rules
            for drule in self.grammar.get_discontinuousrules():
                # create constituents according to the arity of the rule
                constituents = []
                for idx in range(len(drule.rhs)):
                    constituents.append(
                        [c for c, n in nodes if n == drule.rhs[idx]]
                    )
                for pair in product(*constituents):
                    cover = [pos for cs in list(pair) for pos in cs]
                    cover.sort()
                    # cover must not be empty
                    if not cover:
                        continue
                    # positions should be disjoint
                    if len(cover) != len(set(cover)):
                        continue
                    # cover should be discontinuous
                    if max(cover) - min(cover) + 1 <= len(cover):
                        continue
                    # split cover in continuous intervals
                    tmpprev = cover[0] - 1
                    tmpinterval = []
                    tmpcover = [(idx, n)
                                for n, t in enumerate(list(pair))
                                for idx in t]
                    tmpcover.sort(key=itemgetter(0))
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
                        continue
                    # check whether the current item should be pruned
                    subsent = [' '.join([slist[i[0]] for i in tmpi])
                               for tmpi in intervals]
                    subsent.extend([drule.lhs])
                    if pp.ispruned(tuple(subsent)):
                        continue
                    # validate the yield function of the rule
                    run = []
                    for csid in range(len(drule.yf)):
                        subrun = []
                        for child in drule.yf[csid]:
                            if tmpcover:
                                subrun.append(tmpcover[0][1])
                            tmpcover =\
                                list(dropwhile(lambda x: x[1] == child,
                                               tmpcover))
                        run.append(tuple(subrun))
                    # if yf is valid, the run equals the yield function
                    if tuple(run) != drule.yf:
                        continue

                    # add hypernode to hypergraph
                    label = tuple(cover)
                    if not (label in hg.get_nodes()):
                        hg.add_node(Hypernode(label))
                    node = hg.get_nodes()[label]
                    # create new hyperedge
                    succs = list(map(lambda x: hg.get_nodes()[x], list(pair)))
                    he = Hyperedge(
                            drule.lhs, drule.get_prob(),
                            1, node, zip(succs, drule.rhs)
                            )
                    # enqueue the new item if not already visited
                    joinedlists = list(hg.get_edges())
                    joinedlists.extend(list(agenda))
                    if len(list(filter(
                            lambda x: x.get_signature() == he.get_signature(),
                            joinedlists))) > 1:
                        continue
                    agenda.append(he)

            # finally add the item to the hypergraph
            hg.add_edge(item)
        return hg


__all__ = ['Parser']
