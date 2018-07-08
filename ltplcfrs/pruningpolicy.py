"""This module contains the pruning policy and its needed subfunction
deserialize."""

import time
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.linear_model import LinearRegression

from .features import Feature, FeatureItem, handle_item
from .grammar import Grammar


class PruningPolicy(object):
    """The Pruning Policy class represents a collection of pruning decisions
    for nonterminals and their spanning subsentences in a parsing process.

    Attributes
    ----------
    __df : DataFrame
        The raw data set considering the given features.
    __feats : list(Feature)
        The list of used features
    __model : LinearRegression
        The underlying model (classifier) to train.
    __isempty : bool
        The truth value expressing the status of the underlying data set.

    """

    def __init__(self, df=None, rewards=None, featkeys=None,
                 featdict=None, grammar=None):
        """Initialize an instance of a pruning policy.

        Parameters
        ----------
        df : DataFrame
            The raw data set considering the given features.
        featkeys : list(str)
            The list of the used feature keys.
        featdict : dict(str, [str, function, bool])
            The underlying feature dictionary.
        grammar : Grammar
            The grammar.

        """
        # validationsif not isinstance(rewards, list):
        if rewards is not None and not isinstance(rewards, list):
            raise TypeError("rewards must be a list")
        if rewards is not None and\
           not all(isinstance(e, float) for e in rewards):
            raise TypeError("rewards must be a list of floating point numbers")
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("model must be a pandas.DataFrame")
        if (df is not None) and\
           (rewards is not None) and\
           (df.shape[0] != len(rewards)):
            # shape = (rows, columns)
            raise ValueError("reward must have length %i" % self.__df.shape[0])
        if featkeys is not None and (not isinstance(featkeys, list) or
           not all(isinstance(e, str) for e in featkeys)):
            raise TypeError("featkeys must be a list of strings")
        if grammar is not None and not isinstance(grammar, Grammar):
            raise TypeError("grammar must be a ltplcfrs.grammar.Grammar")
        if featdict is not None and not isinstance(featdict, dict):
            raise TypeError("featdict must be a dictionary")
        # assigning
        self.__tr = rewards
        self.__df = df if df is not None else pd.DataFrame()
        self.__model = LinearRegression()
        self.__feats = []
        self.__isempty = False
        if featkeys and featdict and grammar:
            for k in featdict.keys():
                f = featdict[k]
                if k in featkeys:
                    self.__feats.append(Feature(k, f[0], grammar, f[1], f[2]))
        else:
            self.__isempty = True
        if self.__df.shape[0] != 0:
            encoder = ce.OneHotEncoder(cols=list(self.__df))
            df_one_hot = encoder.fit_transform(self.__df)
            self.__model.fit(X=df_one_hot, y=np.array(rewards))
        else:
            self.__isempty = True

    def ispruned(self, item):
        """Check whether an item should be pruned.
        Return true if the pruning policy decides to prune the item,
        otherwise return false.

        Parameters
        ----------
        item : FeatureItem
            The item in question.

        Returns
        -------
        bool
            The truth value expressing whether an item will be pruned.

        """
        if not isinstance(item, FeatureItem):
            raise TypeError("item must be a ltplcfrs.features.FeatureItem")
        if self.__isempty:
            return False
        # never prune leaf items
        if not item.rhs:
            return False
        else:
            sdf = handle_item(item, self.__feats)
            time_enc = time.time()
            encoder = ce.OneHotEncoder(cols=list(self.__df))
            time_fit = time.time()
            encoder.fit_transform(self.__df)
            time_predict = time.time()
            cell = self.__model.predict(X=encoder.transform(sdf))
            time_end = time.time()
            print("encoding: %f - fitting: %f - predicting: %f" %
                  (time_fit - time_enc,
                   time_predict - time_fit,
                   time_end - time_predict))
            # reward(keep) - reward(prune) = cell[0]
            if cell[0] < 0:
                return True
            else:
                return False

    def serialize(self):
        """Serialize the pruning policy into a string.

        Returns
        -------
        str
            Return the serialized pruning policy as a string.

        """
        return pickle.dumps(
            {'featkeys': list(map(Feature.get_id, self.__feats)),
             'df': self.__df.to_msgpack(),
             'tr': self.__tr})


def deserialize(contents, featdict, grammar):
    """Deserialize a string into a pruning policy.

    Parameters
    ----------
    contents : str
        The pruning policy in string format.
    featdict : dict(str, [str, function, bool])
        The underlying feature dictionary.
    grammar : Grammar
        The grammar.

    Returns
    -------
    PruningPolicy
        Return the deserialized pruning policy.

    """
    # validation
    if not isinstance(grammar, Grammar):
        raise TypeError("grammar must be a ltplcfrs.grammar.Grammar")
    if not isinstance(featdict, dict):
        raise TypeError("featdict must be a dictionary")
    # deserialization
    d = pickle.loads(contents)
    return PruningPolicy(pd.read_msgpack(d['df']), d['tr'], d['featkeys'],
                         featdict, grammar)


__all__ = ['PruningPolicy', 'deserialize']
