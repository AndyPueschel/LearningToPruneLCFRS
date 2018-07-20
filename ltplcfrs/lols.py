"""This module contains the lols algorithm and its needed subfunctions.
The implementation is an adaption of the lols algorithm introduced in
'Learning to Prune: Exploring the Frontier of Fast and Accurate Parsing' by
Tim Vieira and Jason Eisner (2017) to satisfy 'prune charts' in form of
hypergraphs."""

from .pruningpolicy import PruningPolicy
from .parse import Parser
from .features import FeatureItem, FeatureCollector

from discodop.tree import Tree


def lols(grammar, corpus, pp=PruningPolicy(), iterations=1, weight=1,
         featkeys=('l', 'sl', 'bwd', 'bwcd', 'ss', 'sw', 'wb', 'wscd')):
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
    featkeys : list(str)
        The list of unique feature keys.

    Returns
    -------
    PruningPolicy
        The trained pruning policy.

    """
    # default values
    pp = pp if pp else PruningPolicy()
    iterations = iterations if iterations is not None else 1
    weight = weight if weight is not None else 1
    if not featkeys:
        featkeys = ('l', 'sl', 'bwd', 'bwcd', 'ss', 'sw', 'wb', 'wscd')
    featkeys = list(featkeys)

    policies = {0: pp}  # i-th pruning policy represents pp after i iterations
    dataset = []  # contains lists of state-reward tuples for each iteration
    rewards = {}  # avg rewards after each iteration for last pruning policy
    parser = Parser(grammar)
    collector = FeatureCollector(grammar, featkeys)
    for i in range(iterations):
        datasubset = []  # contains state-reward tuples
        rewardsum = 0.0  # initialize denominator for avg reward
        for sentence, tree in corpus:
            # roll in
            derivation_graph = parser.parse(' '.join(sentence), policies[i])
            for edge in derivation_graph.get_edges():
                # dont train leaf items
                if not edge.get_successors():
                    continue
                # create vector indices
                r = {}  # two dimensional 'vector' of rewards
                pbit = edge.get_pruningbit()
                nbit = (pbit + 1) % 2
                # create feature item
                lhs = edge.get_nonterminal()
                rhs = [(nt, n.get_label()) for n, nt in edge.get_successors()]
                fitem = FeatureItem(lhs, rhs, sentence)
                # roll out
                r[pbit] = reward(weight, derivation_graph, tree)
                edge.set_pruningbit(nbit)
                r[nbit] = reward(weight, derivation_graph, tree)
                edge.set_pruningbit(pbit)
                # feature item and sentence correspond to a state
                datasubset.append((fitem, r))
            # increase the denominator
            rewardsum += reward(weight, derivation_graph, tree)
        # train
        dataset.append(datasubset)
        policies[i+1] = train(dataset, collector)
        rewards[i] = rewardsum / len(corpus)
    maxidx = max(rewards, key=rewards.get)
    return policies[maxidx]


def train(q, collector):
    """Trains the pruning policy via dataset aggregation.

    Parameters
    ----------
    q : list(list((FeatureItem, dict(int, double))))
        The set of all state reward tuples.
    collector : FeatureCollector
        The feature collector.

    Returns
    -------
    PruningPolicy
        The trained pruning policy.

    """
    dataset = sum(q, []) if q else []
    if not dataset:
        ti, tr = [], []
    else:
        ti, tr = zip(*dataset)
    data, rewards = list(ti), list(tr)
    rewards = [r[1] - r[0] for r in rewards]
    collector.drop_data()
    collector.inject_data(data)
    policy = collector.create_PruningPolicy(rewards)
    return policy


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
    tree = dg.get_tree()
    acc = accuracy(tree, gt)
    run = runtime(dg)
    reward = acc - (l * run)
    return reward


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
