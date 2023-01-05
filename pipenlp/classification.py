from .type_check import *
import copy
import scipy.sparse as sp
import numpy as np


class Classification(object):
    def __init__(self, y: series_type = None):
        self.y = copy.copy(y)
        self.id2label = {}
        self.label2id = {}
        if self.y is not None:
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)

    def fit(self, s):
        return self

    def transform(self, s):
        raise Exception("need to implement")

    def get_params(self) -> dict:
        return dict()

    def set_params(self, params: dict):
        pass

    @staticmethod
    def pd2csr(data: dataframe_type):
        try:
            sparse_matrix = sp.csr_matrix(data.sparse.to_coo().tocsr(), dtype=np.float32)
        except:
            sparse_matrix = sp.csr_matrix(data, dtype=np.float32)
        return sparse_matrix

    @staticmethod
    def pd2dense(data: dataframe_type):
        return data.values


class LGBMClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, max_depth=3, num_boost_round=256):
        Classification.__init__(self, y)
        self.max_depth = max_depth
        self.num_boost_round = num_boost_round
        self.lgb_model = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        import lightgbm as lgb
        s_ = self.pd2csr(s)
        params = {
            'objective': 'multiclass',
            'num_class': self.num_class,
            'max_depth': self.max_depth,
            'verbose': -1
        }
        self.lgb_model = lgb.train(params=params, train_set=lgb.Dataset(data=s_, label=self.y),
                                   num_boost_round=self.num_boost_round)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.lgb_model.predict(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"lgb_model": self.lgb_model, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.lgb_model = params["lgb_model"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class LogisticRegressionClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, multi_class="multinomial", solver="newton-cg", max_iter=1000):
        Classification.__init__(self, y)
        self.multi_class = multi_class
        self.solver = solver
        self.max_iter = max_iter
        self.lr = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.linear_model import LogisticRegression
        s_ = self.pd2csr(s)
        self.lr = LogisticRegression(multi_class=self.multi_class, solver=self.solver, max_iter=self.max_iter)
        self.lr.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.lr.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"lr": self.lr, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.lr = params["lr"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class SVMClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, gamma=2, c=1, kernel="rbf"):
        Classification.__init__(self, y)
        self.gamma = gamma
        self.C = c
        self.kernel = kernel
        self.svm = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.svm import SVC
        s_ = self.pd2csr(s)
        self.svm = SVC(kernel=self.kernel, decision_function_shape="ovo", gamma=self.gamma, C=self.C, probability=True)
        self.svm.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.svm.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"svm": self.svm, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.svm = params["svm"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class DecisionTreeClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, criterion="gini", max_depth=3, min_samples_leaf=16):
        Classification.__init__(self, y)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.tree import DecisionTreeClassifier
        s_ = self.pd2csr(s)
        self.tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
        self.tree.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.tree.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"tree": self.tree, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.tree = params["tree"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class RandomForestClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, n_estimators=128, criterion="gini", max_depth=3, min_samples_leaf=16):
        Classification.__init__(self, y)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.ensemble import RandomForestClassifier
        s_ = self.pd2csr(s)
        self.tree = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
        self.tree.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.tree.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"tree": self.tree, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.tree = params["tree"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class KNeighborsClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None, n_neighbors=5):
        Classification.__init__(self, y)
        self.n_neighbors = n_neighbors
        self.knn = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.neighbors import KNeighborsClassifier
        s_ = self.pd2csr(s)
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.knn.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"knn": self.knn, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.knn = params["knn"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class GaussianNBClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None):
        Classification.__init__(self, y)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import GaussianNB
        s_ = self.pd2dense(s)
        self.nb = GaussianNB()
        self.nb.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2dense(s)
        return pandas.DataFrame(self.nb.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class MultinomialNBClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None):
        Classification.__init__(self, y)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import MultinomialNB
        s_ = self.pd2csr(s)
        self.nb = MultinomialNB()
        self.nb.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.nb.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]


class BernoulliNBClassification(Classification):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |label|
    |good|
    |bad|
    -------------------------
    output type:pandas.dataframe
    output like:
    |good|bad|
    |0.5 |0.5|
    |0.5 |0.5|
    """

    def __init__(self, y: series_type = None):
        Classification.__init__(self, y)
        self.nb = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.naive_bayes import BernoulliNB
        s_ = self.pd2csr(s)
        self.nb = BernoulliNB()
        self.nb.fit(s_, self.y)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        return pandas.DataFrame(self.nb.predict_proba(s),
                                columns=[self.id2label.get(i) for i in range(self.num_class)])

    def get_params(self) -> dict:
        return {"nb": self.nb, "num_class": self.num_class, "id2label": self.id2label}

    def set_params(self, params: dict):
        self.nb = params["nb"]
        self.num_class = params["num_class"]
        self.id2label = params["id2label"]
