"""This module provides the command line interface for the other modules."""

from sys import argv, stdin, stdout, maxsize
from os.path import basename

from discodop.tree import DrawTree, Tree, ImmutableTree
from discodop.treebank import TigerXMLCorpusReader
from discodop.treetransforms import canonicalize

from .pruningpolicy import PruningPolicy, deserialize
from .grammar import Grammar
from .parse import Parser
from .lols import accuracy, lols
from .features import FEATURES


def main():
    """The main function of this module."""

    # some variable initializations
    name = basename(argv[0])
    cmds = dict(COMMANDS)
    stdinput = stdin.readlines()

    # some argument validations
    if len(argv) <= 1:
        print("No command given. Type '%s help' for more info" % name)
    elif argv[1] not in cmds:
        print("Unknown command. Type '%s help' for more info" % name)
    elif len(argv[2:]) > cmds[argv[1]]['maxargs']:
        print("Too many arguments. Type '%s help' for more info" % argv[1])
    elif len(argv[2:]) < cmds[argv[1]]['minargs']:
        print("Not enough arguments. Type '%s help' for more info" % argv[1])

    # call the command with given arguments
    cmds[argv[1]]['cmd'](argv[2:], stdinput)


def train(args, stdinput):
    """Train a given pruning policy.

    Parameters
    ----------
    args : list(str)
        List of arguments: corpus file, number of iterations, weight.
    stdinput : list(str)
        The initial pruning policy as std input.

    """

    # assign the parameter values
    pp = PruningPolicy(stdinput)
    corpus = TigerXMLCorpusReader(args[0], encoding='utf8')
    iterations = int(args[1]) if len(args) >= 2 and args[1] else 1
    weight = float(args[2]) if len(args) >= 3 and args[2] else 1
    slength = int(args[3]) if len(args) >= 4 and args[3] else maxsize
    nsents = int(args[4]) if len(args) >= 5 and args[4] else maxsize
    feats = str(args[5]).split('-') if len(args) >= 6 else None

    # create grammar and corpus
    trees = [ImmutableTree.convert(canonicalize(t))
             for t in list(corpus.trees().values())]
    sentences = list(corpus.sents().values())
    grammar = Grammar(trees, sentences)
    simplecorpus = [(s, _t) for s, _t in list(zip(sentences, trees))
                    if len(s) <= slength]
    if nsents:
        simplecorpus = simplecorpus[:nsents]

    # print the trained pruning policy into the console
    # newpp = lols(grammar, simplecorpus, pp, iterations, weight, feats)
    # stdout.write(str(newpp))
    return lols(grammar, simplecorpus, pp, iterations, weight, feats)


def parse(args, stdinput):
    """Parse a given sentence after inducing a grammar from a given corpus.

    Parameters
    ----------
    args : list(str)
        The list of arguments: corpus file, sentence.
    stdinput : list(str)
        The pruning policy which should be used for the parsing process.

    """

    # assign the parameter values
    corpus = TigerXMLCorpusReader(args[0], encoding='utf8')
    sent = args[1]

    # create grammar and gold trees
    trees = [ImmutableTree.convert(canonicalize(t))
             for t in list(corpus.trees().values())]
    sentences = list(corpus.sents().values())
    grammar = Grammar(trees, sentences)
    goldtrees = [t for s, t in zip(sentences, trees) if ' '.join(s) == sent]

    # create initial pruning policy
    pp = deserialize(stdinput, FEATURES, grammar) if\
        stdinput else PruningPolicy()

    # create derivation tree
    parser = Parser(grammar)
    derivationgraph = parser.parse(sent, pp)
    derivationtree = derivationgraph.get_tree()
    stdout.flush()

    # print results
    if isinstance(derivationtree, Tree):
        # print graphical representation if the sentence could be parsed
        print(derivationtree.pprint())
        drawtree = DrawTree(derivationtree, sent.split())
        print("\n derivation tree: \n" + drawtree.text())
    else:
        # otherwise print a error message
        print(derivationtree)
    if len(goldtrees) > 0:
        # print graphical representation if there is a gold tree
        drawgold = DrawTree(goldtrees[0], sent.split())
        print("\n gold tree: \n" + drawgold.text())
        # print recall if both trees are available
        if isinstance(derivationtree, Tree):
            print("\n recall: %f" % accuracy(derivationtree, goldtrees[0]))


def help(args, stdinput):
    """The help function.

    Parameters
    ----------
    args : None
        Not needed for this function.
    stdinput : None
        Not needed for this function.

    """
    cmds = dict(COMMANDS)
    print(cmds['help']['help'] + '\n' +
          '\n'.join([d['short'] for _c, d in cmds]))


COMMANDS = {
            'train': {'cmd': train,
                      'minargs': 1,
                      'maxargs': 5,
                      'help': ("train: trains a pruning policy and returns it"
                               " via std out. \n"
                               "       Usage: <pp> | train <corpus-file>"
                               " [<i>, <w>, <l>, <n>, <f>] \n"
                               "       <pp>: initial pruning policy via"
                               " stdin \n"
                               "       <corpus-file>: the file path to the"
                               " corpus \n"
                               "       <i>: the number of iterations \n"
                               "       <w>: the accuracy-runtime trade off \n"
                               "       <l>: the maximum sentence length \n"
                               "       <n>: the maximum number of sentences \n"
                               "       <f>: the by '-' seperated features"
                               " such as l, sl, bw, bwc, ..."
                               ),
                      'short': "<pp> | train <corpus-file> [<i>, <w>] \n"},
            'parse': {'cmd': parse,
                      'minargs': 2,
                      'maxargs': 2,
                      'help': ("parse: parses a sentence and returns the"
                               " parse tree and recall. \n"
                               "       Usage: <pp> | parse <corpus-file>"
                               " <s> \n"
                               "       <pp> the pruning policy via stdin \n"
                               "       <corpus-file>: the file path to the"
                               " corpus \n"
                               "       <s>: the sentence to parse \n"
                               ),
                      'short': "<pp> | parse <corpus-file> <s>"},
            'help': {'cmd': help,
                     'minargs': 0,
                     'maxargs': 1,
                     'help': ("help: shows information for commands. \n"
                              "      Usage: <cmd> help \n"
                              "      <cmd>: the command you want information"
                              " aboout \n"
                              ),
                     'short': "<cmd> help"}
            }


if __name__ == "__main__":
    main()


__all__ = ['main', 'parse', 'train']
