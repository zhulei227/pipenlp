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
nlp=PipeNLP()
```

准备`pandas.DataFrame`格式的数据


```python
import pandas as pd
data=pd.read_csv("./data/demo.csv")
data.head(5)
```




|    | text                                                                                                                                                                                  | label   |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------|
|  0 | 动力差                                                                                                                                                                                | 消极    |
|  1 | 油耗很低，操控比较好。第二箱油还没有跑完。油耗显示为5.9了，本人13年12月刚拿的本，跑出这样的油耗很满意了。                                                                             | 积极    |
|  2 | 乘坐舒适性                                                                                                                                                                            | 积极    |
|  3 | 最满意的不止一点：1、车内空间一流，前后排均满足使用需求，后备箱空间相当大；2、外观时尚，珠光白尤其喜欢，看起来也上档次，性价比超高！3、内饰用料在同级别车重相当厚道，很软，手感很好！ | 积极    |
|  4 | 空间大，相对来说舒适性较好，性比价好些。                                                                                                                                              | 积极    |



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




|    | output                                                                                                                                                 |
|---:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 动力差                                                                                                                                                 |
|  1 | 油耗很低操控比较好第二箱油还没有跑完油耗显示为了本人年月刚拿的本跑出这样的油耗很满意了                                                                 |
|  2 | 乘坐舒适性                                                                                                                                             |
|  3 | 最满意的不止一点车内空间一流前后排均满足使用需求后备箱空间相当大外观时尚珠光白尤其喜欢看起来也上档次性价比超高内饰用料在同级别车重相当厚道很软手感很好 |
|  4 | 空间大相对来说舒适性较好性比价好些                                                                                                                     |



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




|    | output                                                                                                                                                                                         |
|---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | 动力 差                                                                                                                                                                                        |
|  1 | 油耗 很 低 操控 比较 好 第二 箱油 还 没有 跑 完 油耗 显示 为了 本人 年 月 刚 拿 的 本 跑 出 这样 的 油耗 很 满意 了                                                                            |
|  2 | 乘坐 舒适性                                                                                                                                                                                    |
|  3 | 最 满意 的 不止 一点 车 内 空间 一流 前后排 均 满足 使用 需求 后备箱 空间 相当 大 外观 时尚 珠光 白 尤其 喜欢 看起来 也 上档次 性价比 超高 内饰 用料 在 同 级别 车重 相当 厚道 很软 手感 很 好 |
|  4 | 空间 大 相对来说 舒适性 较 好性 比价 好些                                                                                                                                                      |


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




|    |         0 |         1 |         2 |         3 |         4 |         5 |         6 |         7 |         8 |         9 |
|---:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
|  0 | 0.0333422 | 0.0333365 | 0.0333361 | 0.0333375 | 0.0333477 | 0.699935  | 0.033347  | 0.0333382 | 0.0333395 | 0.0333399 |
|  1 | 0         | 0.419074  | 0         | 0.555101  | 0         | 0         | 0         | 0         | 0         | 0         |
|  2 | 0.0333399 | 0.0333397 | 0.0333373 | 0.0333425 | 0.0333446 | 0.699946  | 0.0333367 | 0.0333369 | 0.0333382 | 0.0333387 |
|  3 | 0         | 0.159444  | 0         | 0         | 0         | 0         | 0.821498  | 0         | 0         | 0         |
|  4 | 0.0111154 | 0.0111148 | 0.011116  | 0.0111147 | 0.0111152 | 0.0111172 | 0.0111163 | 0.0111155 | 0.0111167 | 0.899958  |



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




|    |        0 |         1 |         2 |           3 |        4 |        5 |         6 |        7 |
|---:|---------:|----------:|----------:|------------:|---------:|---------:|----------:|---------:|
|  0 | 0.521626 | -1.11158  | -0.52375  | -0.0111436  | 0.882553 | 1.28695  | -0.386774 | 0.936366 |
|  1 | 0.44299  | -0.875816 | -0.402543 | -0.013807   | 0.643703 | 1.01134  | -0.319752 | 0.727737 |
|  2 | 0.343862 | -0.639304 | -0.278756 |  0.0214393  | 0.441067 | 0.703636 | -0.224053 | 0.518533 |
|  3 | 0.363618 | -0.727091 | -0.345958 | -0.011569   | 0.538764 | 0.861078 | -0.26783  | 0.611543 |
|  4 | 0.244573 | -0.481686 | -0.217829 |  0.00585448 | 0.361131 | 0.540066 | -0.158017 | 0.373351 |


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




|    |         0 |         1 |
|---:|----------:|----------:|
|  0 | -1.25732  | -0.222364 |
|  1 |  1.07196  |  0.266178 |
|  2 | -1.28851  | -0.212725 |
|  3 |  0.281789 | -0.626023 |
|  4 | -1.29397  | -0.353284 |



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




|    |     积极 |        消极 |
|---:|---------:|------------:|
|  0 | 0.245708 | 0.754292    |
|  1 | 0.913772 | 0.0862285   |
|  2 | 0.4356   | 0.5644      |
|  3 | 0.999868 | 0.000131638 |
|  4 | 0.916361 | 0.0836388   |



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




|    |     积极 |        消极 |
|---:|---------:|------------:|
|  0 | 0.49978  | 0.50022     |
|  1 | 0.802964 | 0.197036    |
|  2 | 0.448628 | 0.551372    |
|  3 | 0.999973 | 2.74161e-05 |
|  4 | 0.779578 | 0.220422    |



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




|    |     积极 |        消极 |
|---:|---------:|------------:|
|  0 | 0.49978  | 0.50022     |
|  1 | 0.802964 | 0.197036    |
|  2 | 0.448628 | 0.551372    |
|  3 | 0.999973 | 2.74161e-05 |
|  4 | 0.779578 | 0.220422    |


## TODO  

- 加入TextCNN、HAT、LSTM+Attention、Bert等更高阶的文本分类模型  
- 支持预训练词向量


```python

```
