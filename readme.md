# Dive into Deep Learning(Pytorch)
Datawhale第10期组队学习活动笔记及个人手写练习代码   
课程页：https://www.boyuai.com/elites/course/cZu18YmweLv10OeV
## Task01 02.12
### 1.线性回归（Linear Regression）
回归问题的目标是在给定D维输入(input)变量x的情况下，预测一个或多个连续目标(target)变量t的值.  

最简单的回归模型（线性函数模型）：y(x, w) = w0 + w1x1 + ... + wDxD   

损失函数：MSE(Mean Square Error)   

优化器：SGD(Stochastic Gradient Descent)

#### 实验
##### Dependencies
- torch
- numpy
- random
- tqdm
##### Dataset
generate by random
##### Code
/src/LinearRegression.py
##### Results
| true_weights | true_bais | dataset_size | batch_size | lr | it_nums | result |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| [2, -3.4] | 4.2 | 1000 | 64 | 0.03 | 100 | [2.0005, -3.4003], 4.2001 |
| [10.2, -33.4, 0.84] | 0.4 | 1000 | 64 | 0.03 | 100 | [10.2002, -33.4000, 0.8397], 0.3998 |

### 2.Softmax与分类模型(Softmax Classification)
分类问题的目标是将输入变量x分到K个离散的类别Ck中的某一类  

分类问题的目标值为离散型   
2分类：t=1表示C1，t=0表示C2；   
K分类(K>2)：one-hot编码；  

Softmax回归：是logistic回归模型在多分类问题上的推广，在多分类问题中，类标签y可以取两个以上的值。softmax运算不改变预测类别输出。   

Softmax分类模型：O=XW+B;Y=Softmax(O)  

损失函数：交叉熵损失函数(CrossEntropyLoss)。最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率（最大似然估计）。  

优化器：SGD(Stochastic Gradient Descent)
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
##### Code
/src/SoftmaxClassification.py
##### Result
| batch_size | lr | it_nums | train_loss | test_loss | test_acc |
| :-----| ----: | :----: | :----: | :----: | :----: |
| 64 | 0.01 | 20 | 0.452 | 0.482 | 0.8335 |

### 3.多层感知机(Multilayer Perceptron)
线性模型只能处理线性问题，即使用多个线性模型进行复合也无法处理非线性问题，MLP在的每一层为一个线性模型，在每一层的结束加上一个非线性激活函数（relu,sigmoid,tanh,...），即可以处理非线性问题。   

单隐含层MLP模型：H=ϕ(XWh+Bh);O=HWo+Bo。其中 ϕ 表示激活函数。
单隐含层MLP+Softmax分类模型：H=ϕ(XWh+Bh);O=HWo+Bo;Y=Softmax(O)   

损失函数：交叉熵损失函数(CrossEntropyLoss)  

优化器：SGD(Stochastic Gradient Descent)

#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
##### Code
/src/MultilayerPerceptron.py
##### Result
| batch_size | lr | it_nums | layer_dims | train_loss | test_loss | test_acc |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| 64 | 0.01 | 20 | [28*28, 256, 128, 10] | 0.349 | 0.403 | 0.855 |
| 64 | 0.01 | 20 | [28*28, 256, 10] | 0.318 | 0.371 | 0.869 |

可以看出，相同超参数设置，mlp比单线性层效果要好，但是继续增加隐含层数量效果不一定正比增加

## Task02 02.13
### 1.文本预处理
文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：

- 读入文本
- 分词，用现有工具进行分词(spaCy和NLTK)
- 建立字典，将每个词映射到一个唯一的索引（index）
- 将文本从词的序列转换为索引的序列，方便输入模型
### 2.语言模型
一段自然语言文本可以看作是一个离散时间序列，给定一个长度为 T 的词的序列 w1,w2,…,wT ，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：P(w1,w2,…,wT)      
词的概率可以通过该词在训练数据集中的相对词频来计算。序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。 n 元语法通过马尔可夫假设简化模型，马尔科夫假设是指一个词的出现只与前面 n 个词相关，即 n 阶马尔可夫链（Markov Chain of Order  n ）

### 3.循环神经网络基础(RNN)
RNN目的是基于当前的输入与过去的输入序列，预测序列的下一个字符。循环神经网络引入一个隐藏变量 H ，用 Ht 表示 H 在时间步 t 的值。 Ht 的计算基于 Xt 和 Ht−1 ，可以认为 Ht 记录了到当前字符为止的序列信息，利用 Ht 对序列的下一个字符进行预测。

RNN:Ht=ϕ(XtWxh+Ht−1Whh+bh);Ot=HtWhq+bq.  

梯度裁剪：循环神经网络中较容易出现梯度衰减或梯度爆炸，这会导致网络几乎无法训练。裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。假设我们把所有模型参数的梯度拼接成一个向量  g ，并设裁剪的阈值是 θ 。裁剪后的梯度的 L2 范数不超过 θ 。  

one-hot编码：
- original: [1,2,3]
- one-hot: [[0,1,0,0],[0,0,1,0],[0,0,0,1]]   

#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset
jaychou_lyrics.txt   
歌词数据集：周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词
##### Code
/src/nlp_tools.py
/src/rnn.py
##### Result

## Task03 02.15
### 1.过拟合、欠拟合及其解决方案
训练误差(training error):训练集上的误差   
泛化误差(generalization error):用测试集上的误差代替   
数据集划分：训练集(train set)/验证集(validate set)/测试集(test set)
其中训练集用来调整模型参数，验证集用来调整超参数   
k折交叉验证(k-folder cross-validation):将数据集分成k等份，每次取一份作为验证集，其余作为训练集，得到k组训练结果，最后对着k组训练结果取平均(k-folder可以降低数据集划分对结果的影响)   
过拟合(overfiting): 训练误差很低但是验证误差很高(数据不够或者参数太多)   
欠拟合(underfiting):训练集误差和验证集误差都很高(参数太少，拟合能力不够)  
解决过拟合的方法：
- 权重衰减(L2范数正则化)  
loss = loss + λ/2n|w|^2   
同时需要修改优化器中权重迭代方式

- DropOut   
当对该隐藏层使用丢弃法时，该层的隐藏单元将有一定概率被丢弃掉。设丢弃概率为 p ，那么有 p 的概率 hi 会被清零，有 1−p 的概率 hi 会除以 1−p 做拉伸。Dropout不改变输入的期望值。

使用MLP对两种方法进行验证
- 模型：MLP
- 优化器：sgd
- 损失函数：交叉熵
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
##### Code
/src/regularization_and_dropout.py
##### Result
| batch_size | lr | it_nums | layer_dims | decay_lamd | dropout_prob | train_loss | test_loss |test_val |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 64 | 0.01 | 20 | [28*28, 256, 10] | 0.0 | 0.0 | 0.318 | 0.371 | 0.869 |
| 64 | 0.01 | 30 | [28*28, 256, 10] | 0.0 | 0.0 | 0.290 | 0.358 | 0.874 |
| 64 | 0.01 | 20 | [28*28, 256, 10] | 0.1 | 0.0 | 0.732 | 0.541 | 0.819 |
| 64 | 0.01 | 20 | [28*28, 256, 10] | 0.0 | 0.1 | 0.332 | 0.369 | 0.869 |
| 64 | 0.01 | 30 | [28*28, 256, 10] | 0.0 | 0.1 | 0.294 | 0.350 | 0.876 |

可以看出，dropout对模型的泛化性能确实有提升，但是权重衰减效果不明显，（权重衰减的notebook原始代码没有修改权重更新的部分）

### 2.梯度消失、梯度爆炸
深度模型有关数值稳定性问题主要有两种，分别为消失(vanishing)和爆炸(explosion)   
当神经网络的层数较多时，模型的数值稳定性容易变差。   

参数初始化：
- Pytorch的默认随机初始化   
- Xavier随机初始化   
权重参数随机采样于均匀分布U(-(6/(a+b))^0.5, (6/(a+b))^0.5),其中a,b分别为该层输入个数与输出个数，它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。   

环境因素：
- 协变量偏移
- 标签偏移
- 概念偏移

### 3.循环神经网络进阶
RNN   
只有一个隐含单元
GRU   
加入一个重置门与更新门   
- 重置门：有助于捕捉时间序列⾥短期的依赖关系；
- 更新门：有助于捕捉时间序列⾥长期的依赖关系；
LSTM   
加入输入门， 遗忘门， 输出门，记忆单元
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset

##### Code
/src/gru_and_lstm.py
##### Result

## Task04 02.16
### 1.机器翻译及相关技术
机器翻译（MT）：将一段文本从一种语言自动翻译为另一种语言，用神经网络解决这个问题通常称为神经机器翻译（NMT）。   
主要特征：输出是单词序列而不是单个单词。 输出序列的长度可能与源序列的长度不同。   
Encoder-Decoder：用来解决输入输出序列长度不等的问题   
Sequence to Sequence模型：   
Beam Search方法： 选择整体分数最高的句子（搜索空间太大）集束搜索
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset

##### Code
/src/seq2seq.py
##### Result

### 2.注意力机制与Seq2seq模型
注意力机制:
- 背景信息经过长序列的传播信息丢失严重
- 解码的目标词语可能只与原输入的部分词语有关

Attention框架： Attention是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）。对于一个query来说，attention layer 会与每一个key计算注意力分数并进行权重的归一化，输出的向量 o 则是value的加权求和，而每个key计算的权重与value一一对应。   
- ai=α(q,ki).
- b1,…,bn=softmax(a1,…,an).
- o=∑bivi

masked_softmax: 解决之前加入padding的序列
Attention的两种计算方法：
- 点乘注意力(The dot product)
- 多层感知机注意力
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset

##### Code
/src/seq2seq_with_attention.py
##### Result
### 3.Transformer
- CNNs 易于并行化，却不适合捕捉变长序列内的依赖关系。
- RNNs 适合捕捉长距离变长序列的依赖，但是却难以实现并行化处理序列。

Transformer模型利用attention机制实现了并行化捕捉序列依赖，并且同时处理序列的每个位置的tokens，上述优势使得Transformer模型在性能优异的同时大大减少了训练时间。
- Transformer blocks
- Add and norm
- Position encoding
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset

##### Code
/src/seq2seq_with_attention.py
##### Result


## Task05 02.17
### 1.卷积神经网络
2D卷积计算公式：   
2D池化计算公式：   
#### 实验
##### Dependencies
- torch
- torchvision
- numpy
- tqdm
##### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
##### Code
/src/cnn.py
##### Result

### 2.LeNet


### 3.AlexNet，NiN, GoogLeNet

## Task06 02.19
### 1.批量归一化和残差网络
- 输入标准化(浅层模型)   
使输入数据各个特征的分布近似
- 批量归一化BatchNormalization(深度模型)   
使整个神经网络在各层的中间输出的数值更稳定
#### (1)对全连接层做批量归一化
    保留通道维，计算第一维的均值和方差
#### (1)对卷积层做批量归一化    
    保留通道维， 计算其余维的均值和方差
### 2.凸优化

### 3.梯度下降
- 一维梯度下降   
- 多维梯度下降   

自适应方法：牛顿法   
随机梯度下降   
小批量随机梯度下降   
动态学习率

## Task07 02.20
### 1.优化算法进阶
### 2.word2vec
(1)Skip-Gram跳字模型
(2)CBOW(continuous bag-of-words)连续词袋模型
### 3.词嵌入进阶
GloVe 全局向量的词嵌入    
- 使用非概率分布的变量  p′ij=xij  和  q'ij=exp(u⊤jvi) ，并对它们取对数   
- 为每个词  wi  增加两个标量模型参数：中心词偏差项  bi  和背景词偏差项  ci ，松弛了概率定义中的规范性
- 将每个损失项的权重  xi  替换成函数  h(xij) ，权重函数  h(x)  是值域在  [0,1]  上的单调递增函数，松弛了中心词重要性与  xi  线性相关的隐含假设
- 用平方损失函数替代了交叉熵损失函数
## Task08 02.21
### 1.文本分类
文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。   
#### 文本情感分类
使用文本情感分类来分析文本作者的情绪。
- 使用循环神经网络进行情感分类   
双向循环神经网络
- 使用卷积神经网络进行情感分类   
1维卷积网络

### 2.数据增强
- 翻转和裁剪/padding
- 颜色抖动
- mixup
- cutmix
- random erasing
- augmix
- autoaugment
### 3.模型微调

## Task09 02.22
### 1.目标检测基础
### 2.图像风格迁移
### 3.图像分类案例1

## Task10 02.23
### 1.图像分类案例2
### 2.GAN
### 3.DCGAN
