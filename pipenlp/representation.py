from .type_check import *
import numpy as np


class RePresentation(object):
    def fit(self, s):
        return self

    def transform(self, s):
        raise Exception("need to implement")

    def get_params(self) -> dict:
        return dict()

    def set_params(self, params: dict):
        pass


class BagOfWords(RePresentation):
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

    def __init__(self):
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
        return self

    def transform(self, s: series_type) -> dataframe_type:
        tf_vectors_csr = self.tf.transform(s)
        return pandas.DataFrame.sparse.from_spmatrix(
            tf_vectors_csr, s.index, self.tf.get_feature_names_out()
        )

    def get_params(self) -> dict:
        return {"tf": self.tf}

    def set_params(self, params: dict):
        self.tf = params["tf"]


class TFIDF(RePresentation):
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

    def __init__(self):
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
        return self

    def transform(self, s: series_type) -> dataframe_type:
        tf_vectors_csr = self.tfidf.transform(s)
        return pandas.DataFrame.sparse.from_spmatrix(
            tf_vectors_csr, s.index, self.tfidf.get_feature_names_out()
        )

    def get_params(self) -> dict:
        return {"tfidf": self.tfidf}

    def set_params(self, params: dict):
        self.tfidf = params["tfidf"]


class PCADecomposition(RePresentation):
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

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(s.values)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        return pandas.DataFrame(self.pca.transform(s.values))

    def get_params(self) -> dict:
        return {"pca": self.pca, "n_components": self.n_components}

    def set_params(self, params: dict):
        self.pca = params["pca"]
        self.n_components = params["n_components"]


class NMFDecomposition(RePresentation):
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

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.nmf = None

    @check_dataframe_type
    def fit(self, s: dataframe_type) -> dataframe_type:
        from sklearn.decomposition import NMF
        self.nmf = NMF(n_components=self.n_components)
        self.nmf.fit(s.values)
        return self

    def transform(self, s: dataframe_type) -> dataframe_type:
        return pandas.DataFrame(self.nmf.transform(s.values))

    def get_params(self) -> dict:
        return {"nmf": self.nmf, "n_components": self.n_components}

    def set_params(self, params: dict):
        self.nmf = params["nmf"]
        self.n_components = params["n_components"]


class LdaTopicModel(RePresentation):
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

    def __init__(self, num_topics=10):
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
        return self

    def transform(self, s: series_type) -> dataframe_type:
        from gensim import matutils
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        vectors = matutils.corpus2dense(self.lda_model[common_corpus], num_terms=self.num_topics).T
        return pandas.DataFrame(vectors)

    def get_params(self) -> dict:
        return {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lda_model": self.lda_model}

    def set_params(self, params: dict):
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lda_model = params["lda_model"]


class LsiTopicModel(RePresentation):
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

    def __init__(self, num_topics=10):
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
        return self

    def transform(self, s: series_type) -> dataframe_type:
        from gensim import matutils
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        common_corpus = [self.common_dictionary.doc2bow(text) for text in texts]
        vectors = matutils.corpus2dense(self.lsi_model[common_corpus], num_terms=self.num_topics).T
        return pandas.DataFrame(vectors)

    def get_params(self) -> dict:
        return {"num_topics": self.num_topics, "common_dictionary": self.common_dictionary, "lsi_model": self.lsi_model}

    def set_params(self, params: dict):
        self.num_topics = params["num_topics"]
        self.common_dictionary = params["common_dictionary"]
        self.lsi_model = params["lsi_model"]


class Word2VecModel(RePresentation):
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

    def __init__(self, embedding_size=16, min_count=5):
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.w2v_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.models import Word2Vec
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.w2v_model = Word2Vec(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
        return self

    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [np.mean(np.asarray([self.w2v_model.wv[word] for word in line if word in self.w2v_model.wv]),
                           axis=0) + np.zeros(shape=(self.embedding_size,))
                   for line in texts]
        return pandas.DataFrame(vectors)

    def get_params(self) -> dict:
        return {"embedding_size": self.embedding_size, "w2v_model": self.w2v_model, "min_count": self.min_count}

    def set_params(self, params: dict):
        self.embedding_size = params["embedding_size"]
        self.w2v_model = params["w2v_model"]
        self.min_count = params["min_count"]


class Doc2VecModel(RePresentation):
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

    def __init__(self, embedding_size=16, min_count=5):
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
        return self

    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [self.d2v_model.infer_vector(line) for line in texts]
        return pandas.DataFrame(vectors)

    def get_params(self) -> dict:
        return {"embedding_size": self.embedding_size, "d2v_model": self.d2v_model, "min_count": self.min_count}

    def set_params(self, params: dict):
        self.embedding_size = params["embedding_size"]
        self.d2v_model = params["d2v_model"]
        self.min_count = params["min_count"]


class FastTextModel(RePresentation):
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

    def __init__(self, embedding_size=16, min_count=5):
        self.min_count = min_count
        self.embedding_size = embedding_size
        self.fasttext_model = None

    @check_series_type
    def fit(self, s: series_type) -> dataframe_type:
        from gensim.models import FastText
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        self.fasttext_model = FastText(sentences=texts, vector_size=self.embedding_size, min_count=self.min_count)
        return self

    def transform(self, s: series_type) -> dataframe_type:
        texts = s.values.tolist()
        texts = [line.split(" ") for line in texts]
        vectors = [np.mean(np.asarray([self.fasttext_model.wv[word]
                                       for word in line if word in self.fasttext_model.wv]), axis=0) + np.zeros(
            shape=(self.embedding_size,))
                   for line in texts]
        return pandas.DataFrame(vectors)

    def get_params(self) -> dict:
        return {"embedding_size": self.embedding_size, "fasttext_model": self.fasttext_model,
                "min_count": self.min_count}

    def set_params(self, params: dict):
        self.embedding_size = params["embedding_size"]
        self.fasttext_model = params["fasttext_model"]
        self.min_count = params["min_count"]
