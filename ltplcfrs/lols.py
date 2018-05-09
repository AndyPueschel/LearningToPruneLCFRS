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
from sys import stdout


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
            print(str(sentence))
            print(tree.pprint())
            stdout.flush()
            derivation_graph = parser.parse(' '.join(sentence), policies[i])
            r = {}  # two dimensional 'vector' of rewards
            for edge in derivation_graph.get_edges():
                pbit = edge.get_pruningbit()
                nbit = (pbit + 1) % 2
                # roll out
                print("roll-out")
                stdout.flush()
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

    # create raw data
    for idx, states in q.items():
        for (s, r) in states:
            signature = list(s[0].get_signature())
            lhs_cover = signature[0][0]
            # split cover in continuous intervals
            tmpprev = lhs_cover[0] - 1
            tmpinterval = []
            intervals = []
            for pos in lhs_cover:
                if pos == tmpprev + 1:
                    tmpinterval.append(pos)
                    tmpprev = pos
                else:
                    intervals.append(tmpinterval)
                    tmpinterval = [pos]
                    tmpprev = pos
            intervals.append(tmpinterval)
            # create sentence
            subsent = [' '.join([s[1][i] for i in tmpi])
                       for tmpi in intervals]
            # colname is a policy item
            subsent.append(s[0].get_nonterminal())
            colname = tuple(subsent)
            if colname not in colnames:
                colnames.append(colname)
            # fill the data
            col = data.get(colname, [np.nan] * (len(q) + 1))
            col[idx] = r[0] - r[1]
            data[colname] = col

    # create initial dataframe
    df = DataFrame([[np.nan] * len(colnames)], columns=colnames)

    # fill dataframe and interpolate
    for idx in range(len(q)):
        for colname in colnames:
            tmp_value = data[colname][idx]
            if not np.isnan(tmp_value):
                df.at[idx, colname] = tmp_value
        tmp_frame = DataFrame([[np.nan] * len(colnames)], columns=colnames)
        df = df.append(tmp_frame, ignore_index=True)
        # possible methods: barycentric, krogh, pchip,
        # spline (order=1 or 3)
        if idx == 0:
            df = df.interpolate(method='linear')
        else:
            df = df.interpolate(method='spline', order=1)

    print(str(df))
    stdout.flush()
    # collect data for new pruning policy
    pp = PruningPolicy()
    pp.mode = Mode.KEEPING
    pp.contents = [i for i in colnames if df[i][len(q)] < 0]
    print(str(pp.contents))
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
    if not isinstance(dt, Tree):
        return 0
    # count the correct nodes with the right cover
    candidates = [(st.label, tuple(sorted(st.leaves())))
                  for st in dt.subtrees()]
    golds = [(st.label, tuple(sorted(st.leaves()))) for st in gt.subtrees()]
    matches = [pair for pair in candidates if pair in golds]

    # calculate measuring metrics
    recall = float(len(matches) / len(golds))
    precision = float(len(matches) / len(candidates))
    f1measure = float((2 * recall * precision) / (recall + precision))
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
