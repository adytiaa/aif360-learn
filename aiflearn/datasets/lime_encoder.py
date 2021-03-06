import sklearn.preprocessing
import numpy as np

from aiflearn.algorithms import Transformer


class LimeEncoder(Transformer):
    """Tranformer for converting aiflearn dataset to LIME dataset and vice versa.

    (LIME - Local Interpretable Model-Agnostic Explanations) [2]_

    See for details/usage:
    https://github.com/marcotcr/lime

    References:
        .. [2] M.T. Ribeiro, S. Singh, and C. Guestrin, '"Why should I trust
           you?" Explaining the predictions of any classifier.'
           https://arxiv.org/pdf/1602.04938v1.pdf
    """

    def __init__(self):
        super(LimeEncoder, self).__init__()

    def fit(self, dataset):
        """Take an aiflearn dataset and save all relevant metadata as well as
        mappings needed to transform/inverse_transform the data between aiflearn
        and lime.

        Args:
            dataset (BinaryLabelDataset): aiflearn dataset

        Returns:
            LimeEncoder: Returns self.
        """
        # ohe = one hot encoding
        self.s_feature_names_with_ohe = dataset.feature_names
        df, df_dict = dataset.convert_to_dataframe(de_dummy_code=True)

        # remove label (class) column
        dfc = df.drop(dataset.label_names[0], axis=1)

        # create list of feature names
        self.s_feature_names = list(dfc.columns)
        # create array of feature values
        self.s_data = dfc.values

        # since categorical features are 1-hot-encoded and their names changed,
        # the set diff gives us the list of categorical features as non-
        # categorical feature names are not changed
        self.s_categorical_features = list(
            set(self.s_feature_names)
            - set(self.s_feature_names_with_ohe))

        self.s_protected_attribute_names = dataset.protected_attribute_names

        # add protected attribute names to the list of categorical features
        self.s_categorical_features = (self.s_categorical_features
                                       + self.s_protected_attribute_names)

        self.s_labels = df[dataset.label_names[0]]  # create labels

        # following 3 lines are not really needed
        # using to create s_class_names
        # can do so manually as well ...array([ 0.,  1.])
        s_le = sklearn.preprocessing.LabelEncoder()
        s_le.fit(self.s_labels)
        # self.s_labels = s_le.transform(self.s_labels)
        self.s_class_names = s_le.classes_

        # convert s_categorical_features to a list of array indexes in
        # s_feature_names corresponding to categorical features
        # (NOTE - does not included protected attributes)
        self.s_categorical_features = [self.s_feature_names.index(x)
                                       for x in self.s_categorical_features]

        # map all the categorical features to numerical values and store the
        # mappings in s_categorical_names
        self.s_categorical_names = {}
        for feature in self.s_categorical_features:
            self.le = sklearn.preprocessing.LabelEncoder()
            self.le.fit(self.s_data[:, feature])
            # self.s_data[:, feature] = le.transform(self.s_data[:, feature])
            self.s_categorical_names[feature] = self.le.classes_

        return self

    def transform(self, aiflearndata):
        """Take aiflearn data array and return data array that is lime encoded
        (numeric array in which categorical features are NOT one-hot-encoded).

        Args:
            aiflearndata (np.ndarray): Dataset features

        Returns:
            np.ndarray: LIME dataset features
        """
        tgtNumRows = aiflearndata.shape[0]
        tgtNumcolumns = len(self.s_feature_names)
        limedata = np.zeros(shape=(tgtNumRows, tgtNumcolumns))

        # non_categorical_features = (list(set(self.s_feature_names) &
        # set(self.s_feature_names_with_ohe)))
        for rw in range(limedata.shape[0]):
            for ind, feature in enumerate(self.s_feature_names):
                if ind in self.s_categorical_features:
                    # tranform the value since categorical feature except if it
                    # is also a protected attribute
                    if feature in self.s_protected_attribute_names:
                        # just copy the value as is
                        limedata[rw, ind] = aiflearndata[
                            rw,
                            self.s_feature_names_with_ohe.index(feature)]
                    else:
                        possible_feature_values = self.s_categorical_names[ind]
                        for indc in range(len(possible_feature_values)):
                            cval = possible_feature_values[indc]
                            colName = feature + "=" + cval
                            if (aiflearndata[rw][
                                    self.s_feature_names_with_ohe.index(
                                        colName)] == 1.0):
                                limedata[rw][ind] = indc
                else:
                    # just copy the value as is
                    limedata[rw, ind] = aiflearndata[
                        rw, self.s_feature_names_with_ohe.index(feature)]

        return limedata

    def inverse_transform(self, limedata):
        """Take data array that is lime encoded (that is, lime-compatible data
        created by this class from a given aiflearn dataset) and return data
        array consistent with the original aiflearn dataset.

        Args:
            limedata (np.ndarray): Dataset features

        Returns:
            np.ndarray: aiflearn dataset features
        """
        tgtNumRows = limedata.shape[0]
        tgtNumcolumns = len(self.s_feature_names_with_ohe)
        aiflearndata = np.zeros(shape=(tgtNumRows, tgtNumcolumns))
        feature_names = self.s_feature_names_with_ohe

        for rw in range(aiflearndata.shape[0]):
            for ind, feature in enumerate(self.s_feature_names):
                # s_categorical_features has list of indexes into
                # s_feature_names for categorical features
                if ind in self.s_categorical_features:
                    if feature in self.s_protected_attribute_names:
                        # just copy the value as is
                        aiflearndata[rw, feature_names.index(feature)] = \
                            limedata[rw, ind]
                    else:
                        # s_categorical_names[ind] has mapping of categorical
                        # to numerical values i.e. limedata[rw, ind] is index
                        # of this array. value is string val
                        new_feature = (feature + '=' +
                                       self.s_categorical_names[ind][
                                           int(limedata[rw, ind])])
                        # categorical feature:
                        aiflearndata[rw, self.s_feature_names_with_ohe.index(
                            new_feature)] = 1.0
                else:  # just copy value
                    aiflearndata[rw, self.s_feature_names_with_ohe.index(
                        feature)] = limedata[rw, ind]

        return aiflearndata
