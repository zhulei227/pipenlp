from .base import *
import numpy as np
import pandas as pd


class BagOfWords(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ---------------------------
    output type:pandas.dataframe
    output like:
    | i |love|eat|apple|china|
    | 1 |  1 | 1 |  1  |  0  |
    | 1 |  1 | 0 |  0  |  1  |
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.tf = None

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from sklearn.feature_extraction.text import CountVectorizer
        self.tf = CountVectorizer(
            max_features=None,
            tokenizer=self.tokenizer_func,
            preprocessor=self.preprocessor_func,
            min_df=1,
            max_df=1.0,
            binary=False,
        )
        self.tf.fit(s)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        tf_vectors_csr = self.tf.transform(s)
        try:
            feature_names = self.tf.get_feature_names()
        except:
            feature_names = self.tf.get_feature_names_out()
        self.output_col_names = feature_names
        return pandas.DataFrame.sparse.from_spmatrix(data=tf_vectors_csr, columns=feature_names)

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict:
        params = PipeObject.get_params(self)
        params.update({"tf": self.tf})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.tf = params["tf"]


class TFIDF(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    -----------------------------
    output type:pandas.dataframe
    output like:
    | i |love|eat|apple|china|
    |0.2|0.3 |0.4| 0.5 |  0  |
    |0.3|0.2 | 0 |  0  | 0.2 |
    """

    def __init__(self, name=None, transform_check_max_number_error=1e-5, skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.tfidf = None

    @staticmethod
    def tokenizer_func(x):
        return x.split(" ")

    @staticmethod
    def preprocessor_func(x):
        return x

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(
            max_features=None,
            tokenizer=self.tokenizer_func,
            preprocessor=self.preprocessor_func,
            min_df=1,
            max_df=1.0,
            binary=False,
        )
        self.tfidf.fit(s)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        tf_vectors_csr = self.tfidf.transform(s)
        try:
            feature_names = self.tfidf.get_feature_names()
        except:
            feature_names = self.tfidf.get_feature_names_out()
        self.output_col_names = feature_names
        return pandas.DataFrame.sparse.from_spmatrix(data=tf_vectors_csr, columns=feature_names)

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"tfidf": self.tfidf})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.tfidf = params["tfidf"]


class PCADecomposition(PipeObject):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    ---------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 |
    |-0.4|0.4|
    |0.2|0.6|
    """

    def __init__(self, n_components=3, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.n_components = n_components
        self.pca = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(s.fillna(0).values)
        self.input_col_names = s.columns.tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.pca.transform(s.fillna(0).values))
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"pca": self.pca, "n_components": self.n_components})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.pca = params["pca"]
        self.n_components = params["n_components"]


class NMFDecomposition(PipeObject):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 |
    |0.4|0.4|
    |0.2|0.6|
    """

    def __init__(self, n_components=3, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.n_components = n_components
        self.nmf = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import NMF
        self.nmf = NMF(n_components=self.n_components)
        self.nmf.fit(s.fillna(0).values)
        self.input_col_names = s.columns.tolist()
        return self

    @check_dataframe_type
    def transform(self, s: dataframe_type) -> dataframe_type:
        result = pandas.DataFrame(self.nmf.transform(s.fillna(0).values))
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        input_dataframe = input_dataframe[self.input_col_names]
        return self.transform(input_dataframe).to_dict("record")[0]

    def get_params(self) -> dict:
        params = PipeObject.get_params(self)
        params.update({"nmf": self.nmf, "n_components": self.n_components})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.nmf = params["nmf"]
        self.n_components = params["n_components"]


class LdaTopicModel(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, num_topics=10, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.num_topics = num_topics
        self.common_dictionary = None
        self.lda_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.ldamulticore import LdaModel
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.common_dictionary = Dictionary(texts)
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        self.lda_model = LdaModel(common_corpus, num_topics=self.num_topics)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        from gensim import matutils
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        vectors = matutils.corpus2dense(self.lda_model[common_corpus], num_terms=self.num_topics).T
        result = pandas.DataFrame(vectors)
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update(
            {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lda_model": self.lda_model})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lda_model = params["lda_model"]


class LsiTopicModel(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, num_topics=10, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.num_topics = num_topics
        self.common_dictionary = None
        self.lsi_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models import LsiModel
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.common_dictionary = Dictionary(texts)
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        self.lsi_model = LsiModel(common_corpus, num_topics=self.num_topics, id2word=self.common_dictionary)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        from gensim import matutils
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        vectors = matutils.corpus2dense(self.lsi_model[common_corpus], num_terms=self.num_topics).T
        result = pandas.DataFrame(vectors)
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update(
            {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lsi_model": self.lsi_model})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lsi_model = params["lsi_model"]


class Word2VecModel(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.w2v_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.models import Word2Vec
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.w2v_model = Word2Vec(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [np.mean(np.asarray([self.w2v_model.wv[word] for word in line if word in self.w2v_model.wv]),
                           axis=0) + np.zeros(shape=(self.embedding_size,))
                   for line in texts]
        result = pandas.DataFrame(vectors)
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"embedding_size": self.embedding_size, "w2v_model": self.w2v_model, "min_count": self.min_count})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.embedding_size = params["embedding_size"]
        self.w2v_model = params["w2v_model"]
        self.min_count = params["min_count"]


class Doc2VecModel(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.d2v_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        self.d2v_model = Doc2Vec(documents, vector_size=self.embedding_size, min_count=self.min_count)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [self.d2v_model.infer_vector(line) for line in texts]
        result = pandas.DataFrame(vectors)
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"embedding_size": self.embedding_size, "d2v_model": self.d2v_model, "min_count": self.min_count})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.embedding_size = params["embedding_size"]
        self.d2v_model = params["d2v_model"]
        self.min_count = params["min_count"]


class FastTextModel(PipeObject):
    """
    input type:pandas.series
    input like:(space separation)
    |input|
    |i love eat apple|
    |i love china|
    ----------------------------
    output type:pandas.dataframe
    output like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    """

    def __init__(self, embedding_size=16, min_count=5, name=None, transform_check_max_number_error=1e-5,
                 skip_check_transform_type=True):
        PipeObject.__init__(self, name, transform_check_max_number_error, skip_check_transform_type)
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.fasttext_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.models import FastText
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.fasttext_model = FastText(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
        self.input_col_names = [s.name]
        return self

    @check_series_type
    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [np.mean(np.asarray([self.fasttext_model.wv[word]
                                       for word in line if word in self.fasttext_model.wv]), axis=0) + np.zeros(
            shape=(self.embedding_size,))
                   for line in texts]
        result = pandas.DataFrame(vectors)
        self.output_col_names = result.columns.tolist()
        return result

    @check_dict_type
    def transform_single(self, s: dict_type):
        col = self.input_col_names[0]
        return self.transform(
            pd.DataFrame({col: [s.get(col)]})[col]).to_dict("record")[0]

    def get_params(self) -> dict_type:
        params = PipeObject.get_params(self)
        params.update({"embedding_size": self.embedding_size, "fasttext_model": self.fasttext_model,
                       "min_count": self.min_count})
        return params

    def set_params(self, params: dict):
        PipeObject.set_params(self, params)
        self.embedding_size = params["embedding_size"]
        self.fasttext_model = params["fasttext_model"]
        self.min_count = params["min_count"]
