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

注意：只安装了部分必要的包，其他依赖包可在使用到相关模块时再自行安装，比如在使用`pipenlp.preprocessing.ExtractJieBaWords`时才会提示安装`pip install jieba`，所以建议先用少量数据检验pipeline流程的依赖包是否完整，在应用到大量数据上

## 使用  

导入`PipeNLP`主程序


```python
from pipenlp import PipeNLP
nlp=PipeNLP()
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
nlp.pipe(RemoveDigits())\
   .pipe(RemovePunctuation())\
   .pipe(RemoveWhitespace())
```
```python
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
```
```python
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
```
```python
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
```
```python
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
      <td>0.033336</td>
      <td>0.033338</td>
      <td>0.033340</td>
      <td>0.033342</td>
      <td>0.033337</td>
      <td>0.033348</td>
      <td>0.033346</td>
      <td>0.033343</td>
      <td>0.033338</td>
      <td>0.699931</td>
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
      <td>0.970957</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.033337</td>
      <td>0.033336</td>
      <td>0.033336</td>
      <td>0.699944</td>
      <td>0.033355</td>
      <td>0.033342</td>
      <td>0.033339</td>
      <td>0.033339</td>
      <td>0.033336</td>
      <td>0.033337</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.978554</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.011115</td>
      <td>0.011116</td>
      <td>0.011115</td>
      <td>0.011116</td>
      <td>0.011116</td>
      <td>0.011116</td>
      <td>0.899960</td>
      <td>0.011115</td>
      <td>0.011115</td>
      <td>0.011116</td>
    </tr>
  </tbody>
</table>
</div>



#### FastText模型


```python
nlp=PipeNLP()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(FastTextModel(embedding_size=8))
```
```python
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
      <td>0.499252</td>
      <td>-1.062278</td>
      <td>-0.498653</td>
      <td>0.006183</td>
      <td>0.841716</td>
      <td>1.210205</td>
      <td>-0.377396</td>
      <td>0.903633</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.453218</td>
      <td>-0.888729</td>
      <td>-0.410743</td>
      <td>0.002836</td>
      <td>0.651204</td>
      <td>1.012078</td>
      <td>-0.332197</td>
      <td>0.746859</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.323486</td>
      <td>-0.590322</td>
      <td>-0.257216</td>
      <td>0.031456</td>
      <td>0.402557</td>
      <td>0.635742</td>
      <td>-0.212270</td>
      <td>0.482930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.366357</td>
      <td>-0.726029</td>
      <td>-0.347917</td>
      <td>0.001929</td>
      <td>0.535537</td>
      <td>0.846559</td>
      <td>-0.273708</td>
      <td>0.617496</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.239123</td>
      <td>-0.466997</td>
      <td>-0.211472</td>
      <td>0.014322</td>
      <td>0.347688</td>
      <td>0.514858</td>
      <td>-0.157679</td>
      <td>0.364920</td>
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
```
```python
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
      <td>-0.222479</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.071957</td>
      <td>0.266118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.288506</td>
      <td>-0.212698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.281789</td>
      <td>-0.625973</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.293974</td>
      <td>-0.353327</td>
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
```
```python
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
```
```python
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
      <td>0.502272</td>
      <td>0.497728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.780132</td>
      <td>0.219868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.452948</td>
      <td>0.547052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999974</td>
      <td>0.000026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.776129</td>
      <td>0.223871</td>
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
```python
nlp.load("nlp.pkl")
```


```python
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
      <td>0.502272</td>
      <td>0.497728</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.780132</td>
      <td>0.219868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.452948</td>
      <td>0.547052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999974</td>
      <td>0.000026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.776129</td>
      <td>0.223871</td>
    </tr>
  </tbody>
</table>
</div>



## TODO  

- 加入TextCNN、Bert更高阶文本分类模型  
- 引入预训练词向量


```python

```
