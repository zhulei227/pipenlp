from .type_check import *
import string


class PreProcessing(object):
    def call(self, s):
        raise Exception("need to implement")

    def fit(self, s):
        return self

    def transform(self, s):
        return self.call(s)

    def get_params(self) -> dict:
        return dict()

    def set_params(self, params: dict):
        pass


class Lower(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |Abc中午|
    |jKK|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc中午|
    |jkk|
    """

    @check_series_type
    def call(self, s):
        return s.str.lower()


class Upper(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |Abc中午|
    |jKK|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |ABC中午|
    |JKK|
    """

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.upper()


class RemoveDigits(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |abc123|
    |j1k2|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc|
    |jk|
    """

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.replace(r"\d+", "")


class ReplaceDigits(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |abc123|
    |j1k2|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abc[d]|
    |j[d]k[d]|
    """

    def __init__(self, symbols="[d]"):
        self.symbols = symbols

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.replace(r"\d+", self.symbols)

    def get_params(self) -> dict:
        return {"symbols": self.symbols}

    def set_params(self, params: dict):
        self.symbols = params["symbols"]


class RemovePunctuation(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |ab,cd;ef|
    |abc!|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abcdef|
    |abc|
    """

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", "")


class ReplacePunctuation(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |ab,cd;ef|
    |abc!|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |ab[p]cd[p]ef|
    |abc[p]|
    """

    def __init__(self, symbols="[p]"):
        self.symbols = symbols

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", self.symbols)

    def get_params(self) -> dict:
        return {"symbols": self.symbols}

    def set_params(self, params: dict):
        self.symbols = params["symbols"]


class RemoveWhitespace(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |ab cd ef|
    |a b c|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |abcdef|
    |abc|
    """

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.str.replace("\xa0", " ").str.split().str.join("")


class RemoveStopWords(PreProcessing):
    """
    stop_words=["ab"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |cd ef|
    |abc|
    """

    def __init__(self, stop_words=None, stop_words_path=None):
        self.stop_words = set()
        if stop_words is not None:
            for word in stop_words:
                self.stop_words.add(word)
        if stop_words_path is not None:
            for line in open(stop_words_path, encoding="utf8"):
                self.stop_words.add(line.strip())

    def apply_function(self, s):
        words = []
        for word in str(s).split(" "):
            if word not in self.stop_words:
                words.append(word)
        return " ".join(words)

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    def get_params(self) -> dict:
        return {"stop_words": self.stop_words}

    def set_params(self, params: dict):
        self.stop_words = params["stop_words"]


class ExtractKeyWords(PreProcessing):
    """
    key_words=["ab","cd"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd|
    |ab|
    """

    def __init__(self, key_words=None, key_words_path=None):
        import ahocorasick
        self.actree = ahocorasick.Automaton()
        if key_words is not None:
            for word in key_words:
                self.actree.add_word(word, word)
        if key_words_path is not None:
            for line in open(key_words_path, encoding="utf8"):
                word = line.strip()
                self.actree.add_word(word, word)
        self.actree.make_automaton()

    def apply_function(self, s):
        words = []
        for i in self.actree.iter(s):
            words.append(i[1])
        return " ".join(words)

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    def get_params(self) -> dict:
        return {"actree": self.actree}

    def set_params(self, params: dict):
        self.actree = params["actree"]


class AppendKeyWords(PreProcessing):
    """
    key_words=["ab","cd"]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd ef ab cd|
    |abc def ab|
    """

    def __init__(self, key_words=None, key_words_path=None):
        import ahocorasick
        self.actree = ahocorasick.Automaton()
        if key_words is not None:
            for word in key_words:
                self.actree.add_word(word, word)
        if key_words_path is not None:
            for line in open(key_words_path, encoding="utf8"):
                word = line.strip()
                self.actree.add_word(word, word)
        self.actree.make_automaton()

    def apply_function(self, s):
        words = []
        for i in self.actree.iter(s):
            words.append(i[1])
        return s + " " + " ".join(words)

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    def get_params(self) -> dict:
        return {"actree": self.actree}

    def set_params(self, params: dict):
        self.actree = params["actree"]


class ExtractChineseWords(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |ab中文cd， e输入f|
    |a中b文c|
    -------------------------
    output type:pandas.series
    output like:
    |output|
    |中文输入|
    |中文|
    """

    @staticmethod
    def apply_function(s):
        import re
        return "".join(re.findall(r'[\u4e00-\u9fa5]', s))

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))


class ExtractNGramWords(PreProcessing):
    """
    demo1:
    n_grams=[2]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |abcd cdef|
    |abcdef|
    -------------------------
    demo2:
    n_grams=[1,2]
    ----------------------------
    input type:pandas.series
    input like:(space separation)
    |input|
    |ab cd ef|
    |abc def|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |ab cd ef abcd cdef|
    |abc def abcdef|
    """

    def __init__(self, n_grams=None):
        if n_grams is not None:
            self.n_grams = n_grams
        else:
            self.n_grams = [2]

    def apply_function(self, s):
        if " " in s:
            s = s.split(" ")
        words = []
        if 1 in self.n_grams:
            for word in s:
                words.append(word)
        if 2 in self.n_grams:
            for i in range(len(s) - 1):
                words.append("".join(s[i:i + 2]))
        if 3 in self.n_grams:
            for i in range(len(s) - 2):
                words.append("".join(s[i:i + 3]))
        return " ".join(words)

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    def get_params(self) -> dict:
        return {"n_grams": self.n_grams}

    def set_params(self, params: dict):
        self.n_grams = params["n_grams"]


class ExtractJieBaWords(PreProcessing):
    """
    input type:pandas.series
    input like:
    |input|
    |北京天安门|
    |北京清华大学|
    -------------------------
    output type:pandas.series
    output like:(space separation)
    |output|
    |北京 天安门|
    |北京 清华大学|
    """

    def __init__(self, cut_all=False):
        self.cut_all = cut_all

    def apply_function(self, s):
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return " ".join(jieba.cut(s, cut_all=self.cut_all))

    @check_series_type
    def call(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    def get_params(self) -> dict:
        return {"cut_all": self.cut_all}

    def set_params(self, params: dict):
        self.cut_all = params["cut_all"]
