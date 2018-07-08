"""This module provides the three classes Feature, FeatureItem and
FeatureCollector which are needed for a feature oriented training.
Furthermore an expandable feature dictionary (FEATURES) and an
item(_to_feature)_handler are contained as well."""

import pandas as pd

from discodop.grammar import ranges

from .grammar import Grammar


class FeatureItem:
    """The FeatureItem class represents an item which will be used for
    training and pruning decisions.

    Attributes
    ----------
    lhs : str
        The lefthand side.
    cover : tuple(int)
        The cover.
    rhs : list((str, tuple(int)))
        The righthand side.
    sentence : list(str)
        The sentence as a list of its words.

    """

    def __init__(self, lhs, rhs, s):
        """Initialize an instance of a feature item.

        Parameters
        ----------
        lhs : str
            The lefthand side.
        rhs : list((str, tuple(int)))
            The righthand side.
        s : list(str)
            The sentence as a list of words.

        """
        # validations
        if not isinstance(lhs, str):
            raise TypeError("he must be a string")
        if not isinstance(rhs, list) or\
           not all(isinstance(e[0], str) and isinstance(e[1], tuple)
           for e in rhs):
            raise TypeError("rhs must be a list of tuple-string pairs")
        if not isinstance(s, list) or\
           not all(isinstance(e, str) for e in s):
            raise TypeError("s must be a list of strings")
        # assigning
        self.lhs = lhs
        cover = [pos for _n, cs in rhs for pos in cs]
        cover.sort()
        self.cover = tuple(cover)
        self.rhs = rhs
        self.sentence = s

    def __str__(self):
        slhs = "(" + self.lhs + ", " + str(self.cover) + ")"
        lrhs = ["(" + n + ", " + str(c) + ")" for n, c in self.rhs]
        srhs = " ".join(lrhs)
        return slhs + " -> " + srhs


class Feature:
    """The Feature class represents a feature to use for training and pruning.
    Its main purpose is providing the feature function.

    Attributes
    ----------
    __id : str
        The unique id.
    __name : str
        The feature name.
    __grammar : Grammar
        The grammar.
    __func : function(Feature, FeatureItem)
        The feature function.
    __multiple : bool
        The truth value expressing the quantity of outputs of the feature func.

    """

    def __init__(self, id, name, grammar, func, multiple=False):
        """Initialize an instance of a feature.

        Parameters
        ----------
        id : str
            The unique id.
        name : str
            The feature name.
        grammar : Grammar
            The grammar.
        func : function(Feature, FeatureItem)
            The feature function.
        multiple : bool
            The truth value for having multiple outputs in the feature func.

        """
        # validations
        if not isinstance(id, str):
            raise TypeError("id must be a string")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(grammar, Grammar):
            raise TypeError("grammar must be a ltplcfrs.grammar.Grammar")
        if not isinstance(multiple, bool):
            raise TypeError("multiple must be a bool")
        if not callable(func):
            raise TypeError("func must be a function")
        # assigning
        self.__id = id
        self.__name = name
        self.__grammar = grammar
        self.__func = func
        self.__multiple = multiple

    def get_id(self):
        """Return the unique id of the feature.

        Returns
        -------
        str
            The unique feature id.

        """
        return self.__id

    def ismultiple(self):
        """Check wheter the feature function provides more than one result.

        Returns
        -------
        bool
            The truth value.

        """
        return self.__multiple

    def get_name(self):
        """Return the name of the feature.

        Returns
        -------
        str
            The feature name.

        """
        return self.__name

    def get_grammar(self):
        """Return the grammar of the feature.

        Returns
        -------
        Grammar
            The grammar.

        """
        return self.__grammar

    def func(self, item):
        """Call the feature function according to its implementation.

        Parameters
        ----------
        item : FeatureItem
            The feature item.

        Returns
        -------
        obj
            Return the result of the feature function.

        """
        if not isinstance(item, FeatureItem):
            raise TypeError("item must be a ltplcfrs.features.FeatureItem")
        return self.__func(self, item)


def feature_label(that, item):
    """Use the nonterminal symbol as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    str
        The nonterminal of the feature item.

    """
    return item.lhs


def feature_sent_length(that, item):
    """Use the sentence length as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    int
        The sentence length.

    """
    return len(item.sentence)


def feature_boundary_words(that, item):
    """Use the two outmost boundary words as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The two boundary words.

    """
    cover = item.cover
    return [item.sentence[cover[0]], item.sentence[cover[-1]]]


def feature_boundary_words_disc(that, item):
    """Use the two outmost boundary words for each span in the cover
    as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The two times the max fanout of boundary words.

    """
    result = []
    for span in ranges(item.cover):
        fake_item = FeatureItem('dummy', [], item.sentence)
        fake_item.sentence = item.sentence
        fake_item.cover = tuple(span)
        result.extend(feature_boundary_words(that, fake_item))
    result.extend(["#none#"] * 2 * that.get_grammar().max_fanout)
    return result[:2 * that.get_grammar().max_fanout]


def feature_boundary_words_conjunction(that, item):
    """Use the different conjunctions of the two outmost boundary words at
    positions i and k as a feature: (i-1, i), (k, k+1), (i-1, k+1), (i, k).

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The four different conjunctions of the boundary words.

    """
    cover = item.cover
    s = item.sentence[cover[0]]
    e = item.sentence[cover[-1]]
    if cover[0] == 0:
        p = "#out_of_sentence#"
    else:
        p = item.sentence[cover[0] - 1]
    if cover[-1] == len(item.sentence) - 1:
        n = "#out_of_sentence#"
    else:
        n = item.sentence[cover[-1] + 1]
    return ["{0} {1}".format(p, s),
            "{0} {1}".format(e, n),
            "{0} {1}".format(p, n),
            "{0} {1}".format(s, e)]


def feature_boundary_words_conjunction_disc(that, item):
    """Use the different conjunctions of the two outmost boundary words at
    positions i and k for each continuous span in the cover as a feature:
    (i-1, i), (k, k+1), (i-1, k+1), (i, k).

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The different conjunctions of the boundary words.

    """
    result = []
    for span in ranges(item.cover):
        fake_item = FeatureItem('dummy', [], item.sentence)
        fake_item.sentence = item.sentence
        fake_item.cover = tuple(span)
        result.extend(feature_boundary_words_conjunction(that, fake_item))
    result.extend(["#none#"] * 4 * that.get_grammar().max_fanout)
    return result[:4 * that.get_grammar().max_fanout]


def feature_span_size(that, item):
    """Use the span size (number of covered positions) as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    int
        The span size.

    """
    return len(item.cover)


def feature_span_width(that, item):
    """Use the span width (first position of span to last position of span)
    as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    int
        The span width.

    """
    cover = item.cover
    return cover[-1] - cover[0] + 1


def feature_span_number(that, item):
    """Use the number of spans in the cover as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    int
        The number of spans in the cover.

    """
    return len(ranges(item.cover))


def feature_width_bucket(that, item):
    """Use the categorisized (bucket) width as a feature.

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    str
        The width bucket.

    """
    cover = item.cover
    width = cover[-1] - cover[0] + 1
    if width <= 2:
        return "2"
    elif width == 3:
        return "3"
    elif width == 4:
        return "4"
    elif width == 5:
        return "5"
    elif width <= 10:
        return "6 - 10"
    elif width <= 20:
        return "11 - 20"
    else:
        return "> 20"


def feature_word_shape_conjunction(that, item):
    """Use the different conjunctions of word shapes of the two outmost
    boundary words at positions i and k as a feature:
    (i-1, i), (k, k+1), (i-1, k+1), (i, k).

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The four different conjunct word shapes of the boundary words.

    """
    cover = item.cover
    sent = item.sentence
    psen = []
    for idx in [cover[0] - 1, cover[0], cover[-1], cover[-1] + 1]:
        if idx < 0 or idx >= len(sent):
            psen.append("nothing")
            continue
        word = sent[idx]
        if len(word.lstrip('-')) > 0 and word.lstrip('-')[0].isdigit():
            psen.append("numeric")
        elif word.isalpha() and word[0].isupper():
            psen.append("uppercase")
        elif word.isalpha() and word.islower():
            psen.append("lowercase")
        else:
            psen.append("special")
    return ["{0} {1}".format(psen[0], psen[1]),
            "{0} {1}".format(psen[2], psen[3]),
            "{0} {1}".format(psen[0], psen[3]),
            "{0} {1}".format(psen[1], psen[2])]


def feature_word_shape_conjunction_disc(that, item):
    """Use the different conjunctions of word shapes of the two outmost
    boundary words at positions i and k for each span in the cover
    as a feature: (i-1, i), (k, k+1), (i-1, k+1), (i, k).

    Parameters
    ----------
    that : Feature
        The feature from where this function is called from.
    item : FeatureItem
        The feature item.

    Returns
    -------
    list(str)
        The different conjunct word shapes of the boundary words.

    """
    result = []
    for span in ranges(item.cover):
        fake_item = FeatureItem('dummy', [], item.sentence)
        fake_item.sentence = item.sentence
        fake_item.cover = tuple(span)
        result.extend(feature_word_shape_conjunction(that, fake_item))
    result.extend(["#none#"] * 4 * that.get_grammar().max_fanout)
    return result[:4 * that.get_grammar().max_fanout]


def handle_item(item, feats):
    """Create a data set row for a item and given features.

    Parameters
    ----------
    item : FeatureItem
        The feature item.
    feats : list(Feature)
        The list of features.

    Returns
    -------
    DataFrame
        The data set with a single row.

    """
    # validations
    if not isinstance(item, FeatureItem):
        raise TypeError("item must be a ltplcfrs.features.FeatureItem")
    if not isinstance(feats, list) or\
       not all(isinstance(e, Feature) for e in feats):
        raise TypeError("feats must be a list of ltplcfrs.features.Feature")
    # creating data frame
    sdf = pd.DataFrame()
    for feat in feats:
        if feat.ismultiple():
            for i, r in list(enumerate(feat.func(item))):
                colname = feat.get_name() + "_" + repr(i)
                sdf[colname] = pd.Series(r)
        else:
            sdf[feat.get_name()] = pd.Series(feat.func(item))
    return sdf


FEATURES = {
    'l':    ["label",
             feature_label,
             False],
    'sl':   ["sentence_length",
             feature_sent_length,
             False],
    'bw':   ["boundary_words",
             feature_boundary_words,
             True],
    'bwc':  ["boundary_words_conjunction",
             feature_boundary_words_conjunction,
             True],
    'ss':   ["span_size",
             feature_span_size,
             False],
    'sw':   ["span_width",
             feature_span_width,
             False],
    'wb':   ["width_bucket",
             feature_width_bucket,
             False],
    'wsc':  ["word_shape_conjunction",
             feature_word_shape_conjunction,
             True],
    'bwd':  ["boundary_words_disc",
             feature_boundary_words_disc,
             True],
    'bwcd': ["boundary_words_conjunction_disc",
             feature_boundary_words_conjunction_disc,
             True],
    'wscd': ["word_shape_conjunction_disc",
             feature_word_shape_conjunction_disc,
             True],
    'sn':   ["span_number",
             feature_span_number,
             False]
}


class FeatureCollector:
    """The FeatureCollector class represents a collector for selected features
    and a creator for a resulting pruning policy (classifier).

    Attributes
    ----------
    __grammar : Grammar
        The grammar.
    __feats : list(Feature)
        The list of used features.
    __df : DataFrame
        The set of trainings data.
    __isempty : bool
        The state of the internal set of trainings data.
    __featdict : dic(str, [str, function, bool])
        The underlying feature directory.

    """

    def __init__(self, grammar, featkeys, featdict=FEATURES):
        """Initialize an instance of a feature collector.

        Parameters
        ----------
        grammar : grammar
            The grammar.
        featkeys : list(str)
            The list of used unique feature keys.
        featdict : dict(str, [str, function, bool])
            The underlying feature dictionary.

        """
        # validations
        if not isinstance(grammar, Grammar):
            raise TypeError("grammar must be a ltplcfrs.grammar.Grammar")
        if not isinstance(featkeys, list) or\
           not all(isinstance(e, str) for e in featkeys):
            raise TypeError("featkeys must be a list of strings")
        if not isinstance(featdict, dict):
            raise TypeError("featdict must be a dictionary")
        # assigning
        self.__grammar = grammar
        self.__feats = []
        self.__df = pd.DataFrame()
        self.__isempty = True
        self.__featdict = featdict
        for k in featdict.keys():
            f = featdict[k]
            if k in featkeys:
                self.__feats.append(Feature(k, f[0], grammar, f[1], f[2]))

    def inject_data(self, td):
        """Inject (add) trainings data into the feature collector.

        Parameters
        ----------
        td : list(FeatureItem)
            The list of trainings data.

        """
        # validation
        if not isinstance(td, list):
            raise TypeError("trainings data must be a list")
        if not all(isinstance(i, FeatureItem) for i in td):
            raise TypeError("trainings data must contain "
                            "ltplcfrs.features.FeatureItem")
        # injection
        for item in td:
            sdf = handle_item(item, self.__feats)
            self.__df = self.__df.append(sdf, ignore_index=True)
        self.__isempty = False

    def drop_data(self):
        """Drop the trainings data.

        """
        self.__df = pd.DataFrame()
        self.__isempty = True

    def create_PruningPolicy(self, rewards):
        """Create a pruning policy according to trainings data.

        Returns
        -------
        PruningPolicy
            The resulting pruning policy.

        """
        if self.__isempty:
            raise AttributeError("trainings data are missing")
        # create policy
        from .pruningpolicy import PruningPolicy
        return PruningPolicy(self.__df, rewards,
                             list(map(Feature.get_id, self.__feats)),
                             self.__featdict, self.__grammar)


__all__ = ['Feature', 'FeatureCollector', 'FeatureItem', 'FEATURES',
           'handle_item']
