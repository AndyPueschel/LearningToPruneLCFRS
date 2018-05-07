"""This module contains the lols algorithm and its needed subfunctions.
The implementation is an adaption of the lols algorithm introduced in
'Learning to Prune: Exploring the Frontier of Fast and Accurate Parsing' by
Tim Vieira and Jason Eisner (2017) to satisfy 'prune charts' in form of
hypergraphs."""

from .pruningpolicy import PruningPolicy, Mode
from .parse import Parser

from discodop.tree import Tree

import numpy as np
from pandas import DataFrame


def lols(grammar, corpus, pp=PruningPolicy(), iterations=1, weight=1):
    """The Locally Optimal Learning to Search algorithm.

    Parameters
    ----------
    grammar : Grammar
        The grammar.
    corpus : list((list(str), Tree))
        The corpus.
    pp : PruningPolicy
        The initial pruning policy.
    iterations : int
        The number of iterations.
    weight : double
        The accuracy-runtime trade off.

    Returns
    -------
    PruningPolicy
        The trained pruning policy.

    """
    policies = {0: pp}  # i-th pruning policy represents pp after i iterations
    dataset = {}  # contains lists of state-reward tuples for each iteration
    rewards = {}  # avg rewards after each iteration
    parser = Parser(grammar)
    for i in range(iterations):
        dataset[i] = []
        rewardsum = 0  # initialize denominator for avg reward
        for sentence, tree in corpus:
            # roll in
            derivation_graph = parser.parse(sentence, policies[i])
            r = {}  # two dimensional 'vector' of rewards
            for edge in derivation_graph.get_edges():
                pbit = edge.get_pruningbit()
                nbit = (pbit + 1) % 2
                # roll out
                r[pbit] = reward(weight, derivation_graph, tree)
                edge.set_pruningbit(nbit)
                r[nbit] = reward(weight, derivation_graph, tree)
                edge.set_pruningbit(pbit)
                # edge signature and sentence correspond to a state
                dataset[i].append(((edge, sentence), r))
            # increase the denominator
            rewardsum += reward(weight, derivation_graph, tree)
        # train
        policies[i+1] = train(dataset)
        rewards[i] = rewardsum / len(corpus)
    maxidx = max(rewards, key=rewards.get)
    return policies[maxidx]


def train(q):
    """Trains the pruning policy via dataset aggregation.

    Parameters
    ----------
    q : dict(int, list(((Hyperedge, list(str)), dict(int, double))))
        The set of all state reward tuples.

    Returns
    -------
    PruningPolicy
        The trained pruning policy.

    """
    colnames = []
    data = {}
    for idx, states in q.items():
        for (s, r) in states:
            signature = list(s[0].get_signature()).sort()
            # split cover in continuous intervals
            tmpprev = signature[0] - 1
            tmpinterval = []
            intervals = []
            for pos in signature:
                if pos == tmpprev + 1:
                    tmpinterval.append(pos)
                else:
                    intervals.append(tmpinterval)
                    tmpinterval = [pos]
                    tmpprev = pos
            intervals.append(tmpinterval)
            # create sentence
            subsent = [' '.join([s[1][i] for i in tmpi])
                       for tmpi in intervals]
            # colname is a policy item
            colname = tuple(subsent.extend([s[1]]))
            if colname not in colnames:
                colnames.append(colname)
            # fill the data
            col = data.get(colname, [np.nan] * (len(q) + 1))
            col[idx] = r[0] - r[1]
            data[colname] = col
    # collect data for new pruning policy
    # possible methods: barycentric, krogh, pchip,
    # spline (order=1 or 3)
    df = DataFrame(data).interpolate(method='spline', order=1)
    pp = PruningPolicy()
    pp.mode = Mode.KEEPING
    pp.contents = [i for i in colnames if df[len(q), i] < 0]
    return pp


def reward(l, dg, gt):
    """Return the reward for a given derivation graph, gold tree and
    accuracy-runtime trade off factor.

    Parameters
    ----------
    l : double
        The weight (trade off factor) for the runtime.
    dg : Hypergraph
        The derivation graph.
    gt : Tree
        The gold tree.

    Returns
    -------
    double
        The reward.

    """
    return accuracy(dg.get_tree(), gt) - l * runtime(dg)


def accuracy(dt, gt):
    """Calculates the F1 measure out of labeled recall and labeled precision
    of a derivation tree for a given gold tree.

    Parameters
    ----------
    dt : Tree
        The derivation tree.
    gt : Tree
        The gold tree.

    Returns
    -------
    double
        The accuracy.

    """
    # return the yield of a given tree
    def ntcover(tree):
        if isinstance(tree, int):
            return tree
        elif isinstance(tree, Tree):
            return [pos for cover in list(map(ntcover, tree)) for pos in cover]
        else:
            raise TypeError("A tree must consist of int or tree: %s" % tree)

    # count the correct nodes with the right cover
    candidates = [(st.label, tuple(ntcover(st).sort()))
                  for st in dt.subtrees()]
    golds = [(st.label, tuple(ntcover(st).sort())) for st in gt.subtrees()]
    matches = [pair for pair in candidates if pair in golds]

    # calculate measuring metrics
    recall = len(matches) / len(golds)
    precision = len(matches) / len(candidates)
    f1measure = (2 * recall * precision) / (recall + precision)
    return f1measure


def runtime(dg):
    """Calculate the runtimes of a parsing process according to the hypergraph.

    Parameters
    ----------
    dg : Hypergraph
        The derivation graph.

    Returns
    -------
    int
        The runtime.

    """
    result = 0
    for edge in dg.get_edges():
        if edge.get_pruningbit() != 0:
            result += 1
    return result


__all__ = ['lols']
