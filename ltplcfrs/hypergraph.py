"""This module contains the Hypergraph class and its needed sub classes
Hypernode and Hyperedge."""

from queue import Queue
from sys import stdout

from discodop.tree import Tree, escape


class Hypergraph(object):
    """The Hypergraph class acts as the chart in the parsing process.

    Attributes
    ----------
    sentence : list(str)
        The sentence as a list of words.
    roots : list(Hypernode)
        The list of hypernodes in the graph without predecessors.
    leaves : list(Hypernode)
        The list of hypernodes in the graph without successors.
    nodes : dict(tuple(int), Hypernode)
        The collection of hypernodes mapped to their corresponding cover.
    edges : list(Hyperedge)
        The list of hyperedges (items) in this graph.

    """

    def __init__(self, sentence):
        """Initialize an instance of a hypergraph.

        Parameters
        ----------
        sentence : str
            The sentence.

        """
        self.sentence = list(map(lambda s: escape(s), sentence.split()))
        self.roots = []
        self.leaves = []
        self.nodes = {}
        self.edges = []

        # create initial leaf nodes
        for pos in range(len(self.sentence)):
            n = Hypernode(tuple([pos]))
            self.add_node(n)

    def add_node(self, node):
        """Add a new hypernode to the hypergraph.

        Parameters
        ----------
        node : Hypernode
            The new hypernode.

        """
        self.nodes[node.get_label()] = node
        self.roots.append(node)
        self.leaves.append(node)

    def add_edge(self, edge):
        """Add a new hyperedge to the hypergraph.

        Parameters
        ----------
        edge : Hyperedge
            The new hyperedge.

        """
        self.edges.append(edge)
        for node, _nt in edge.get_successors():
            if node in self.roots:
                self.roots.remove(node)
        edge.get_predecessor().update_witness(edge.get_nonterminal())
        if (edge.get_predecessor() in self.leaves) and \
           (len(list(edge.get_successors())) > 0):
            self.leaves.remove(edge.get_predecessor())

    def get_edges(self):
        """Return the hyperedges.

        Returns
        -------
        list(Hyperedge)
            The list of all hyperedges in this graph.

        """
        return self.edges

    def get_nodes(self):
        """Return the hypernodes.

        Returns
        -------
        list(Hypernode)
            The list of all hypernodes in this graph.

        """
        return self.nodes

    def get_tree(self):
        """Return the derivation tree with the largest probability.

        Returns
        -------
        Tree
            The highest scoring derivation tree.

        """
        print("\ncreate tree...")
        root = tuple(range(len(self.sentence)))
        if root in self.nodes:
            node = self.nodes[root]
            if node.get_witnesses():
                print("possible witnesses:")
                print(node.get_witnesses())
                maxvalue = ("None", (None, 0))
                for k, v in node.get_witnesses().items():
                    if maxvalue[1][1] < v[1]:
                        maxvalue = (k, v)
                try:
                    st = node.get_subtree(maxvalue[0])
                    return st
                except ValueError as ve:
                    return str(ve)
            else:
                return "node: %s has no witnesses" % str(node.get_label())
        else:
            return "The sentence: \"%s\" could not be parsed" %\
                   ' '.join(self.sentence)

    def get_sentence(self):
        """Return the sentence.

        Returns
        -------
        list(string)
            The list of words in the sentence.

        """
        return self.sentence


class Hyperedge(object):
    """The hyperedge class contains the info for every derviation step.

    Attributes
    ----------
    nonterminal : str
        The nonterminal.
    probability : double
        The probability (weight) of this edge.
    pruningbit : int
        The pruning decision (0 or 1) for this edge.
    predecessor : Hypernode
        The hypernode where this hyperedge is pointing to.
    successors : list((Hypernode, str))
        The hypernodes with their corresponding nonterminals.

    """

    def __init__(self, nt, prob, pb, pre, suc=[]):
        """Initialize an instance of a hyperedge.

        Parameters
        ----------
        nt  : str
            The nonterminal.
        prob : double
            The probability of the rule.
        pb : int
            The pruning bit.
        pre : Hypernode
            The predeccessing node`.
        suc : list((Hypernode, str))
            The successing hypernodes paired with their
            corresponding nonterminals.

        """
        # assign parameter values
        self.nonterminal = nt
        self.predecessor = pre
        self.pruningbit = pb
        self.probability = prob
        self.successors = suc
        # update pre and sucs
        if isinstance(suc, list):
            for node, _nt in suc:
                node.add_out(self)
        pre.add_in(self)

    def get_nonterminal(self):
        """Return the nonterminal symbol.

        Returns
        -------
        str
            The nonterminal symbol on the lhs of a rule.

        """
        return self.nonterminal

    def get_predecessor(self):
        """Return the single hypernode.

        Returns
        -------
        Hypernode
            The predeccessing hypernode.

        """
        return self.predecessor

    def get_successors(self):
        """Return the hypernodes.

        Returns
        -------
        list((Hypernode, str))
            The list of successing hypernodes.

        """
        return self.successors

    def get_pruningbit(self):
        """Return the pruning bit for the item.

        Returns
        -------
        int
            The pruning bit.

        """
        return self.pruningbit

    def get_probability(self):
        """Return the probability.

        Returns
        -------
        double
            The probability of the rule.

        """
        return self.probability

    def set_pruningbit(self, pb):
        """Change the pruning bit according to the
        change propagation algorithm. An item is represented by a hyperedge.

        Parameters
        ----------
        pb : int
            The new value of the pruning bit.

        """
        # use a binary classifier
        if pb not in [0, 1]:
            raise ValueError("invalid input: %f - use 0 or 1 instead" % pb)
        else:
            self.pruningbit = pb
            print("set pruning bit for %s to: %i" % (str(self), pb))
        # change propagation.
        updated = []  # list of already updated node-nonterminal tuples
        q = Queue()  # queue of (node, nonterminal) tuples
        q.put((self.predecessor, self.nonterminal))
        while not q.empty():
            tmp_node, tmp_nt = q.get()
            # check whether the item is already updated
            if (tmp_node, tmp_nt) in updated:
                continue
            print("ingoing edges: %s" %
                  str(["(%s, %s) -> %s" %
                      (ie.get_nonterminal(),
                       ie.get_predecessor().get_label(),
                       " ".join(["(%s, %s)" % (nt, str(n.get_label()))
                                for n, nt in ie.get_successors()])
                       )
                      for ie in tmp_node.get_ins()])
                  )
            print("\nold witnesses for %s: %s" %
                  (str(tmp_node.get_label()), str(tmp_node.witness)))
            updated.append((tmp_node, tmp_nt))
            # check whether the change will propagate to the upper nodes
            old_prob = tmp_node.get_witness(tmp_nt)[1]
            print("old prob for %s: %f" %
                  (str(tmp_node.get_witness(tmp_nt)[0]), old_prob))
            tmp_node.update_witness(tmp_nt)
            new_prob = tmp_node.get_witness(tmp_nt)[1]
            print("new witnesses for %s: %s" %
                  (str(tmp_node.get_label()), str(tmp_node.witness)))
            print("new prob for %s: %f" %
                  (str(tmp_node.get_witness(tmp_nt)[0]), new_prob))
            if old_prob == new_prob:
                continue
            # propagate change to the upper nodes
            print("outgoing edges: %s" %
                  str(["%s:%s -> %s" %
                      (oe.get_nonterminal(),
                       str(oe.get_predecessor().get_label()),
                       " ".join([n for _e, n in oe.get_successors()])
                       ) for oe in tmp_node.get_outs()]))
            for out_edge in tmp_node.get_outs():
                pre_node = out_edge.get_predecessor()
                pre_nt = out_edge.get_nonterminal()
                q.put((pre_node, pre_nt))

    def get_signature(self):
        """Return the signature of the hyperedge.
        A signature has the form:
        ((lhs_cover, lhs_nonterminal),
         (rhs1_cover, rhs1_nonterminal),
         ...
        )

        Returns
        -------
        tuple(tuple(tuple(int), str))
            The signature of the hyperedge.

        """
        lhs = [(self.get_predecessor().get_label(), self.nonterminal)]
        rhs = [(c.get_label(), n) for c, n in self.successors]
        result = list(lhs)
        result.extend(list(rhs))
        return tuple(result)


class Hypernode(object):
    """The hypernode class representing an item in the hypergraph.

    Attributes
    ----------
    label : tuple(int)
        The cover of the hypernode as the label.
    witness : dict(str, (Hyperedge, double))
        The witnesses (ingoing hyperedge) for each nonterminal with their prob.
    outs : list(Hyperedge)
        The list of outgoing hyperedges.
    ins : list(Hyperdge)
        The list of ingoing hyperedges.

    """

    def __init__(self, label):
        """Initialize an instance of a hypernode.

        Parameters
        ----------
        label : str
            The covered sentence positions as the label.

        """
        self.label = label
        self.witness = {}
        self.outs = []
        self.ins = []

    def update_witness(self, nt):
        """Update the current node in respect to its witnesses.

        Parameters
        ----------
        nt : str
            The nonterminal corresponding to the witness.

        It is recommended to call this function after changing the nodes
        incoming hyperedges.

        """
        if nt in self.witness:
            del self.witness[nt]
        for edge in self.ins:
            if edge.get_nonterminal() != nt:
                continue
            temp_prod = edge.get_pruningbit() * edge.get_probability()
            if not edge.get_successors():
                if self.get_witness(nt)[1] < temp_prod:
                    self.witness[nt] = (edge, temp_prod)
                continue
            for node, n in edge.get_successors():
                sub_prod = node.get_witness(n)[1] * temp_prod
                if self.get_witness(nt)[1] < sub_prod:
                    self.witness[nt] = (edge, sub_prod)

    def is_leaf(self):
        """Check whether the hypernode is a leaf.

        Returns
        -------
        bool
            The truth value.

        """
        if not self.ins or len(self.ins) == 0:
            return True
        elif not [hn for he in self.ins for hn, _nt in he.get_successors()]:
            return True
        else:
            return False

    def is_root(self):
        """Check whether the hypernode is a root.

        Returns
        -------
        bool
            The truth value.

        """
        if not self.outs:
            return True
        else:
            return False

    def get_witness(self, nt):
        """Return the hyperedge and the probability which is responsible
        for the hypernodes value at a given nonterminal.
        Return Nothing if there exists no such entry.

        Parameters
        ----------
        nt : str
            The nonterminal corresponding to the witness.

        Returns
        -------
        (Hyperdge, double)
            The witnesses with their corresponding probability.
            If no such witness exists for the given nonterminal,
            the (None, 0) tuple will be returned instead.

        """
        return self.witness.get(nt, (None, 0))

    def get_witnesses(self):
        """Return every witness for each possible nonterminal.

        Returns
        -------
        dict(str((Hyperedge, double)))
            The witnesses for each nonterminal.

        """
        return self.witness

    def get_ins(self):
        """Return the ingoing hyperedges.

        Returns
        -------
        list(Hyperedge)
            The list of ingoing hyperedges.

        """
        return self.ins

    def get_outs(self):
        """Return the outgoing hyperedges.

        Returns
        -------
        list(Hyperedge)
            The list of outgoing hyperedges.

        """
        return self.outs

    def add_in(self, i):
        """Add an ingoing hyperedge to the hypernode.

        Parameters
        ----------
        i : Hyperedge
            The ingoing hyperedge.

        """
        self.ins.append(i)
        self.update_witness(i.get_nonterminal())

    def add_out(self, o):
        """Add an outgoing hyperedge to the hypernode.

        Parameters
        ----------
        o : Hyperedge
            The outgoing hyperedge.

        """
        self.outs.append(o)

    def get_label(self):
        """Return the label of this hypernode.

        Returns
        -------
        str
            The covered sentence positions by this hypernode.

        """
        return self.label

    def get_subtree(self, nt):
        """Return a derivation subtree.

        Parameters
        ----------
        nt : str
            The nonterminal to start with.

        Returns
        -------
        Tree
            The tree with the current node as root.

        """
        edge = self.get_witness(nt)[0]
        if edge is None:
            raise ValueError("There is no witness for %s" % nt)
        if not edge.get_successors():
            stdout.flush()
            return Tree(edge.get_nonterminal(), [self.get_label()[0]])
        else:
            s = edge.get_successors()
            print("successors of %s:" % edge.get_nonterminal())
            print(s)
            return Tree(edge.get_nonterminal(),
                        [t.get_subtree(n) for t, n in s])


__all__ = ['Hypergraph', 'Hyperedge', 'Hypernode']
