import re
import string

from .base import *


class SeriesPipeObject(PipeObject):
    def __init__(self, name=None, transform_check_max_number_error=1e-5):
        PipeObject.__init__(self, name, transform_check_max_number_error)

    @check_series_type
    def fit(self, s):
        self.input_col_names = [s.name]
        self.output_col_names = [s.name]
        return self


class Lower(SeriesPipeObject):
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
    def transform(self, s):
        return s.str.lower()

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        return {self.output_col_names[0]: str(s.get(self.input_col_names[0])).lower()}


class Upper(SeriesPipeObject):
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
    def transform(self, s):
        return s.str.upper()

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        return {self.output_col_names[0]: str(s.get(self.input_col_names[0])).upper()}


class RemoveDigits(SeriesPipeObject):
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
    def transform(self, s: series_type) -> series_type:
        return s.str.replace(r"\d+", "")

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        text = re.sub(r"\d+", "", str(s.get(self.input_col_names[0])))
        return {self.output_col_names[0]: text}


class ReplaceDigits(SeriesPipeObject):
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

    def __init__(self, symbols="[d]", name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
        self.symbols = symbols

    @check_series_type
    def transform(self, s: series_type) -> series_type:
        return s.str.replace(r"\d+", self.symbols)

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        text = re.sub(r"\d+", self.symbols, str(s.get(self.input_col_names[0])))
        return {self.output_col_names[0]: text}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"symbols": self.symbols})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.symbols = params["symbols"]


class RemovePunctuation(SeriesPipeObject):
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
    def transform(self, s: series_type) -> series_type:
        return s.str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", "")

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        puns = rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+"
        text = re.sub(puns, "", str(s.get(self.input_col_names[0])))
        return {self.output_col_names[0]: text}


class ReplacePunctuation(SeriesPipeObject):
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

    def __init__(self, symbols="[p]", name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
        self.symbols = symbols

    @check_series_type
    def transform(self, s: series_type) -> series_type:
        return s.str.replace(rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+", self.symbols)

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        puns = rf"([{string.punctuation}+'，。！（）“‘？：；】【、'])+"
        text = re.sub(puns, self.symbols, str(s.get(self.input_col_names[0])))
        return {self.output_col_names[0]: text}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"symbols": self.symbols})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.symbols = params["symbols"]


class RemoveWhitespace(SeriesPipeObject):
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
    def transform(self, s: series_type) -> series_type:
        return s.str.replace("\xa0", " ").str.split().str.join("")

    @check_dict_type
    def transform_single(self, s: dict_type) -> dict_type:
        text = re.sub(r"\xa0", " ", str(s.get(self.input_col_names[0])))
        text = "".join(text.split())
        return {self.output_col_names[0]: text}


class RemoveStopWords(SeriesPipeObject):
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

    def __init__(self, stop_words=None, stop_words_path=None, name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
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
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"stop_words": self.stop_words})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.stop_words = params["stop_words"]


class ExtractKeyWords(SeriesPipeObject):
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

    def __init__(self, key_words=None, key_words_path=None, name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
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
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"actree": self.actree})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.actree = params["actree"]


class AppendKeyWords(SeriesPipeObject):
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

    def __init__(self, key_words=None, key_words_path=None, name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
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
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"actree": self.actree})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.actree = params["actree"]


class ExtractChineseWords(SeriesPipeObject):
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

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    @check_series_type
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))


class ExtractNGramWords(SeriesPipeObject):
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

    def __init__(self, n_grams=None, name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
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
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"n_grams": self.n_grams})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.n_grams = params["n_grams"]


class ExtractJieBaWords(SeriesPipeObject):
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

    def __init__(self, cut_all=False, name=None, transform_check_max_number_error=1e-5):
        SeriesPipeObject.__init__(self, name, transform_check_max_number_error)
        self.cut_all = cut_all

    def apply_function(self, s):
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)
        return " ".join(jieba.cut(s, cut_all=self.cut_all))

    @check_series_type
    def transform(self, s: series_type) -> series_type:
        return s.apply(lambda x: self.apply_function(x))

    @check_dict_type
    def transform_single(self, s: dict_type):
        return {self.output_col_names[0]: self.apply_function(s.get(self.input_col_names[0]))}

    def get_params(self) -> dict:
        params = SeriesPipeObject.get_params(self)
        params.update({"cut_all": self.cut_all})
        return params

    def set_params(self, params: dict):
        SeriesPipeObject.set_params(self, params)
        self.cut_all = params["cut_all"]
