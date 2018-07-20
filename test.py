from ltplcfrs.cli import parse, train
from ltplcfrs.grammar import Grammar
from ltplcfrs.parse import Parser
# from ltplcfrs.pruningpolicy import deserialize

from discodop.treebank import TigerXMLCorpusReader


def test_parse_read():
    ppfile = open("data/policy.txt", "rb")
    pp = ppfile.read()
    ppfile.close()
    file = "data/tiger.1-1000.xml"
    # sent = "Was bewirkt Ihrer Ansicht nach ein solches Verhalten ?"
    sent = "Nun werden sie umworben ."
    parse([file, sent], pp)


def test_parse():
    file = "data/tiger.1-1000.utf8.xml"
    # sent = "Was bewirkt Ihrer Ansicht nach ein solches Verhalten ?"
    sent = "Nun werden sie umworben ."
    parse([file, sent], None)


def test_train():
    file = "data/tiger.1-1000.utf8.xml"
    iterations = 3
    weigth = 0.0001
    slength = 5
    snumber = None
    pp = train([file, iterations, weigth, slength, snumber], None)
    ppfile = open("data/policy.txt", "wb")
    ppfile.write(pp.serialize())
    ppfile.close()


def test_graph():
    file = "data/tiger.1-1000.utf8.xml"
    sent = "Dabei wurde niemand verletzt ."
    corpus = TigerXMLCorpusReader(file, encoding='utf8')
    trees = corpus.trees().values()
    sents = corpus.sents().values()
    grammar = Grammar(trees, sents)
    parser = Parser(grammar)
    graph = parser.parse(sent)
    return graph
    # Nun werden sie umworben . disc
    # Die Story geht so :
    # Dabei wurde niemand verletzt .
    # Dann werde man weitersehen . s1731
    # Vichy-Polizeichef angeklagt


if __name__ == '__main__':
    test_train()


__all_ = ['test_parse', 'test_graph']
