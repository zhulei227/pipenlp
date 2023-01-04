import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import jieba
import os
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 分词
def tokenizer(text):
    return [word for word in jieba.lcut(text) if word not in stop_words]


# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt', encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


stop_words = get_stop_words()  # 加载停用词表
text = data.Field(sequential=True,
                  lower=True,
                  tokenize=tokenizer,
                  stop_words=stop_words)
label = data.Field(sequential=False)

train, val = data.TabularDataset.splits(
    path='data/',
    skip_header=True,
    train='train.tsv',
    validation='validation.tsv',
    format='tsv',
    fields=[('index', None), ('label', label), ('text', text)],
)

text.build_vocab(train, val, vectors=Vectors(name='data/myvector.vector'))  # 加入测试集的vertor
label.build_vocab(train, val)

embedding_dim = text.vocab.vectors.size()[-1]
vectors = text.vocab.vectors
batch_size = 128
train_iter, val_iter = data.Iterator.splits(
    (train, val),
    sort_key=lambda x: len(x.text),
    batch_sizes=(batch_size, len(val)),  # 训练集设置batch_size,验证集整个集合用于测试
)

vocab_size = len(text.vocab)
label_num = len(label.vocab)


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding之后的shape: torch.Size([200, 8, 300])
        self.word_embeddings = self.word_embeddings.from_pretrained(vectors, freeze=False)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embedding_dim,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embedding_dim, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
net = TextCNN(vocab_size, embedding_dim, kernel_sizes, num_channels)
print(net)


def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch_idx, batch in enumerate(train_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  # X转置 y下标从0开始
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_iter):
            X, y = batch.text, batch.label
            X = X.permute(1, 0)
            y.data.sub_(1)  # X转置 y下标从0开始
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型
                if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, val_iter, net, loss, optimizer, num_epochs)
