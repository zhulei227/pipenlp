# pipenlp  
  
## 介绍   
`pipenlp`包可以方便地将NLP任务构建为`Pipline`任务流，目前主要包含的功能有：  
- 数据清洗，关键词提取等：pipenlp.preprocessing
- 文本特征提取，包括bow,tfidf等传统模型；lda,lsi等主题模型；fastext,word2vec等词向量模型；pca,nmf等特征降维模型：pipenlp.representation
- 文本分类，包括lgbm决策树、logistic回归、svm等传统机器学习模型：pipenlp.classification  

## 安装
```bash
pip install git+https://github.com/zhulei227/pipenlp
```  

## 使用  

导入`PipeNLP`主程序


```python
from pipenlp import PipeNLP
```

准备`pandas.DataFrame`格式的数据


```python
import pandas as pd
data=pd.read_csv("./data/demo.csv")
data.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>动力差</td>
      <td>消极</td>
    </tr>
    <tr>
      <th>1</th>
      <td>油耗很低，操控比较好。第二箱油还没有跑完。油耗显示为5.9了，本人13年12月刚拿的本，跑出...</td>
      <td>积极</td>
    </tr>
    <tr>
      <th>2</th>
      <td>乘坐舒适性</td>
      <td>积极</td>
    </tr>
    <tr>
      <th>3</th>
      <td>最满意的不止一点：1、车内空间一流，前后排均满足使用需求，后备箱空间相当大；2、外观时尚，珠...</td>
      <td>积极</td>
    </tr>
    <tr>
      <th>4</th>
      <td>空间大，相对来说舒适性较好，性比价好些。</td>
      <td>积极</td>
    </tr>
  </tbody>
</table>
</div>



### 数据清洗


```python
from pipenlp.preprocessing import *
nlp=PipeNLP()
nlp.pipe(RemoveDigits())\
   .pipe(RemovePunctuation())\
   .pipe(RemoveWhitespace())

data["output"]=nlp.fit(data["text"]).transform(data["text"])
data[["output"]].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>动力差</td>
    </tr>
    <tr>
      <th>1</th>
      <td>油耗很低操控比较好第二箱油还没有跑完油耗显示为了本人年月刚拿的本跑出这样的油耗很满意了</td>
    </tr>
    <tr>
      <th>2</th>
      <td>乘坐舒适性</td>
    </tr>
    <tr>
      <th>3</th>
      <td>最满意的不止一点车内空间一流前后排均满足使用需求后备箱空间相当大外观时尚珠光白尤其喜欢看起来...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>空间大相对来说舒适性较好性比价好些</td>
    </tr>
  </tbody>
</table>
</div>



### 分词  
默认空格表示分割


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())

data["output"]=nlp.fit(data["text"]).transform(data["text"]).head(5)
data[["output"]].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>动力 差</td>
    </tr>
    <tr>
      <th>1</th>
      <td>油耗 很 低 操控 比较 好 第二 箱油 还 没有 跑 完 油耗 显示 为了 本人 年 月 ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>乘坐 舒适性</td>
    </tr>
    <tr>
      <th>3</th>
      <td>最 满意 的 不止 一点 车 内 空间 一流 前后排 均 满足 使用 需求 后备箱 空间 相...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>空间 大 相对来说 舒适性 较 好性 比价 好些</td>
    </tr>
  </tbody>
</table>
</div>



### 文本特征提取  
#### BOW词袋模型


```python
from pipenlp.representation import *
```


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>一</th>
      <th>一下</th>
      <th>一个</th>
      <th>一个劲</th>
      <th>一个月</th>
      <th>一个舒服</th>
      <th>一二</th>
      <th>一些</th>
      <th>一体</th>
      <th>一停</th>
      <th>...</th>
      <th>黄色</th>
      <th>黑</th>
      <th>黑屏</th>
      <th>黑底</th>
      <th>黑烟</th>
      <th>黑白</th>
      <th>黑色</th>
      <th>默认</th>
      <th>鼓包</th>
      <th>齐全</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3865 columns</p>
</div>



#### LDA主题模型


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(LdaTopicModel(num_topics=10))

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.033340</td>
      <td>0.699931</td>
      <td>0.033340</td>
      <td>0.033339</td>
      <td>0.033345</td>
      <td>0.033339</td>
      <td>0.033336</td>
      <td>0.033347</td>
      <td>0.033343</td>
      <td>0.033338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.970954</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.033336</td>
      <td>0.033340</td>
      <td>0.699950</td>
      <td>0.033337</td>
      <td>0.033336</td>
      <td>0.033341</td>
      <td>0.033336</td>
      <td>0.033347</td>
      <td>0.033336</td>
      <td>0.033342</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.978558</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.011114</td>
      <td>0.011114</td>
      <td>0.899973</td>
      <td>0.011115</td>
      <td>0.011114</td>
      <td>0.011114</td>
      <td>0.011114</td>
      <td>0.011114</td>
      <td>0.011114</td>
      <td>0.011114</td>
    </tr>
  </tbody>
</table>
</div>



#### Word2Vec模型


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(Word2VecModel(embedding_size=8))

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.661387</td>
      <td>-0.205723</td>
      <td>0.258319</td>
      <td>0.536984</td>
      <td>0.449666</td>
      <td>0.042449</td>
      <td>1.769084</td>
      <td>0.686457</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.643488</td>
      <td>-0.274599</td>
      <td>0.326613</td>
      <td>0.609752</td>
      <td>0.457684</td>
      <td>-0.092639</td>
      <td>1.953447</td>
      <td>0.827511</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.279103</td>
      <td>-0.064463</td>
      <td>0.215476</td>
      <td>0.347268</td>
      <td>0.171509</td>
      <td>-0.069043</td>
      <td>0.954895</td>
      <td>0.398952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.541652</td>
      <td>-0.224891</td>
      <td>0.234527</td>
      <td>0.451804</td>
      <td>0.338954</td>
      <td>-0.057171</td>
      <td>1.587017</td>
      <td>0.670156</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.476938</td>
      <td>-0.239309</td>
      <td>0.246525</td>
      <td>0.455723</td>
      <td>0.267393</td>
      <td>-0.052079</td>
      <td>1.387308</td>
      <td>0.579978</td>
    </tr>
  </tbody>
</table>
</div>



#### FastText模型


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(Word2VecModel(embedding_size=8))

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.604970</td>
      <td>-0.282136</td>
      <td>0.254526</td>
      <td>0.583515</td>
      <td>0.509314</td>
      <td>0.033652</td>
      <td>1.754926</td>
      <td>0.643264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.583390</td>
      <td>-0.364780</td>
      <td>0.325500</td>
      <td>0.668123</td>
      <td>0.530914</td>
      <td>-0.105864</td>
      <td>1.951512</td>
      <td>0.782000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.248199</td>
      <td>-0.099972</td>
      <td>0.212815</td>
      <td>0.366762</td>
      <td>0.198890</td>
      <td>-0.073193</td>
      <td>0.935753</td>
      <td>0.373854</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.493239</td>
      <td>-0.295528</td>
      <td>0.233387</td>
      <td>0.496731</td>
      <td>0.396241</td>
      <td>-0.067167</td>
      <td>1.582285</td>
      <td>0.633446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.431377</td>
      <td>-0.301842</td>
      <td>0.244642</td>
      <td>0.493892</td>
      <td>0.317640</td>
      <td>-0.061102</td>
      <td>1.376675</td>
      <td>0.545068</td>
    </tr>
  </tbody>
</table>
</div>



#### PCA降维


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(PCADecomposition(n_components=2))

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.257324</td>
      <td>-0.222434</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.071957</td>
      <td>0.266467</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.288506</td>
      <td>-0.212718</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.281789</td>
      <td>-0.626062</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.293974</td>
      <td>-0.353279</td>
    </tr>
  </tbody>
</table>
</div>



### 文本分类
#### LGBM


```python
from pipenlp.classification import *
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(LGBMClassification(y=data["label"]))

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.245708</td>
      <td>0.754292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.913772</td>
      <td>0.086228</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.435600</td>
      <td>0.564400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999868</td>
      <td>0.000132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.916361</td>
      <td>0.083639</td>
    </tr>
  </tbody>
</table>
</div>



### Logistic回归


```python
from pipenlp.classification import *
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(PCADecomposition(n_components=8))\
   .pipe(LogisticRegressionClassification(y=data["label"])) 

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.507574</td>
      <td>0.492426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.806157</td>
      <td>0.193843</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.449968</td>
      <td>0.550032</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999967</td>
      <td>0.000034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.780798</td>
      <td>0.219202</td>
    </tr>
  </tbody>
</table>
</div>



### 模型持久化
#### 保存


```python
nlp.save("nlp.pkl")
```

#### 加载
由于只保留了模型参数，所以需要重新声明模型结构信息(参数无需传入)


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(PCADecomposition())\
   .pipe(LogisticRegressionClassification())
```




    <pipenlp.pipenlp.PipeNLP at 0x20ce742a188>




```python
nlp.load("nlp.pkl")
nlp.transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.507574</td>
      <td>0.492426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.806157</td>
      <td>0.193843</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.449968</td>
      <td>0.550032</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999967</td>
      <td>0.000034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.780798</td>
      <td>0.219202</td>
    </tr>
  </tbody>
</table>
</div>

## 高阶用法
#### 自定义pipe模块  
自需要实现一个包含了fit,transform,get_params和set_params这四个函数的类即可：  
- fit:用于拟合模块中参数
- transform:利用fit阶段拟合的参数去转换数据
- get_params/set_params:主要用于模型的持久化   

比如如下实现了一个对逐列归一化的模块


```python
import numpy as np
import scipy.stats as ss
class Normalization(object):
    def __init__(self, normal_range=100, normal_type="cdf", std_range=10):
        """
        :param normal_range:
        :param normal_type: cdf,range
        :param std_range:
        """
        self.normal_range = normal_range
        self.normal_type = normal_type
        self.std_range = std_range
        self.mean_std = dict()

    def fit(self, s):
        if self.normal_type == "cdf":
            for col in s.columns:
                col_value = s[col]
                mean = np.median(col_value)
                std = np.std(col_value) * self.std_range
                self.mean_std[col] = (mean, std)
        return self

    def transform(self, s):
        if self.normal_type == "cdf":
            for col in s.columns:
                if col in self.mean_std:
                    s[col] = np.round(
                        ss.norm.cdf((s[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in s.columns:
                if col in self.mean_std:
                    s[col] = self.normal_range * s[col]
        return s

    def get_params(self) -> dict:
        return {"mean_std": self.mean_std, "normal_range": self.normal_range, "normal_type": self.normal_type}

    def set_params(self, params: dict):
        self.mean_std = params["mean_std"]
        self.normal_range = params["normal_range"]
        self.normal_type = params["normal_type"]
```


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(PCADecomposition(n_components=8))\
   .pipe(LogisticRegressionClassification(y=data["label"]))\
   .pipe(Normalization())

nlp.fit(data["text"]).transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.44</td>
      <td>49.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53.94</td>
      <td>46.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.74</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.22</td>
      <td>43.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.58</td>
      <td>46.42</td>
    </tr>
  </tbody>
</table>
</div>



### pipenlp的拆分与合并
PipeNLP的pipe对象也可以是一个PipeNLP，所以我们以将过长的pipeline拆分为多个pipeline，分别fit后再进行组合，避免后面流程的pipe模块更新又要重新fit前面的pipe模块


```python
nlp1=PipeNLP()
nlp1.pipe(ExtractChineseWords())\
    .pipe(ExtractJieBaWords())\
    .pipe(BagOfWords())\
    .pipe(PCADecomposition(n_components=8))

pca_df=nlp1.fit(data["text"]).transform(data["text"])
pca_df.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.257324</td>
      <td>-0.222444</td>
      <td>-0.217656</td>
      <td>-0.118004</td>
      <td>-0.204318</td>
      <td>-0.156744</td>
      <td>-0.009275</td>
      <td>0.088116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.071957</td>
      <td>0.266230</td>
      <td>1.563062</td>
      <td>0.070491</td>
      <td>-1.205419</td>
      <td>0.808515</td>
      <td>-0.552603</td>
      <td>-0.990944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.288506</td>
      <td>-0.212731</td>
      <td>-0.270413</td>
      <td>-0.095124</td>
      <td>-0.098725</td>
      <td>-0.014406</td>
      <td>-0.003268</td>
      <td>0.106673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.281789</td>
      <td>-0.626053</td>
      <td>2.274090</td>
      <td>0.573257</td>
      <td>1.360892</td>
      <td>-0.242684</td>
      <td>-0.069548</td>
      <td>0.656460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.293974</td>
      <td>-0.353290</td>
      <td>0.239912</td>
      <td>0.013726</td>
      <td>0.844865</td>
      <td>-0.353885</td>
      <td>0.112731</td>
      <td>-0.263181</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlp2=PipeNLP()
nlp2.pipe(LogisticRegressionClassification(y=data["label"]))\
    .pipe(Normalization())

nlp2.fit(pca_df).transform(pca_df).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.45</td>
      <td>49.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53.90</td>
      <td>46.10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.74</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.27</td>
      <td>43.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.67</td>
      <td>46.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlp_combine=PipeNLP()
nlp_combine.pipe(nlp1).pipe(nlp2)

nlp_combine.transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.45</td>
      <td>49.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53.90</td>
      <td>46.10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.74</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.27</td>
      <td>43.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.67</td>
      <td>46.33</td>
    </tr>
  </tbody>
</table>
</div>



#### 持久化


```python
nlp_combine.save("nlp_combine.pkl")
```


```python
nlp1=PipeNLP()
nlp1.pipe(ExtractChineseWords())\
    .pipe(ExtractJieBaWords())\
    .pipe(BagOfWords())\
    .pipe(PCADecomposition())

nlp2=PipeNLP()
nlp2.pipe(LogisticRegressionClassification())\
    .pipe(Normalization())

nlp_combine=PipeNLP()
nlp_combine.pipe(nlp1).pipe(nlp2)
nlp_combine.load("nlp_combine.pkl")
```


```python
nlp_combine.transform(data["text"]).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>积极</th>
      <th>消极</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.45</td>
      <td>49.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53.90</td>
      <td>46.10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.74</td>
      <td>50.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.27</td>
      <td>43.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53.67</td>
      <td>46.33</td>
    </tr>
  </tbody>
</table>
</div>

## TODO  

- 加入TextCNN、HAT、LSTM+Attention、Bert更高阶模型  
- 支持预训练词向量


```python

```
