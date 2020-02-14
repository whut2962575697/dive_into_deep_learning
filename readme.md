# Task01 02.12
## 1.线性回归（Linear Regression）
回归问题的目标是在给定D维输入(input)变量x的情况下，预测一个或多个连续目标(target)变量t的值.  

最简单的回归模型（线性函数模型）：y(x, w) = w0 + w1x1 + ... + wDxD   

损失函数：MSE(Mean Square Error)   

优化器：SGD(Stochastic Gradient Descent)

### 实验
#### Dependencies
- torch
- numpy
- random
- tqdm
#### Dataset
generate by random
#### Code
/src/LinearRegression.py
#### Results
| true_weights | true_bais | dataset_size | batch_size | lr | it_nums | result |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| [2, -3.4] | 4.2 | 1000 | 64 | 0.03 | 100 | [2.0005, -3.4003], 4.2001 |
| [10.2, -33.4, 0.84] | 0.4 | 1000 | 64 | 0.03 | 100 | [10.2002, -33.4000, 0.8397], 0.3998 |

## 2.Softmax与分类模型(Softmax Classification)
分类问题的目标是将输入变量x分到K个离散的类别Ck中的某一类  

分类问题的目标值为离散型   
2分类：t=1表示C1，t=0表示C2；   
K分类(K>2)：one-hot编码；  

Softmax回归：是logistic回归模型在多分类问题上的推广，在多分类问题中，类标签y可以取两个以上的值。softmax运算不改变预测类别输出。   

Softmax分类模型：O=XW+B;Y=Softmax(O)  

损失函数：交叉熵损失函数(CrossEntropyLoss)。最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率（最大似然估计）。  

优化器：SGD(Stochastic Gradient Descent)
### 实验
#### Dependencies
- torch
- torchvision
- numpy
- tqdm
#### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
#### Code
/src/SoftmaxClassification.py
#### Result
| batch_size | lr | it_nums | train_loss | test_loss | test_acc |
| :-----| ----: | :----: | :----: | :----: | :----: |
| 64 | 0.01 | 20 | 0.452 | 0.482 | 0.8335 |

## 3.多层感知机(Multilayer Perceptron)
线性模型只能处理线性问题，即使用多个线性模型进行复合也无法处理非线性问题，MLP在的每一层为一个线性模型，在每一层的结束加上一个非线性激活函数（relu,sigmoid,tanh,...），即可以处理非线性问题。   

单隐含层MLP模型：H=ϕ(XWh+Bh);O=HWo+Bo。其中 ϕ 表示激活函数。
单隐含层MLP+Softmax分类模型：H=ϕ(XWh+Bh);O=HWo+Bo;Y=Softmax(O)   

损失函数：交叉熵损失函数(CrossEntropyLoss)  

优化器：SGD(Stochastic Gradient Descent)

### 实验
#### Dependencies
- torch
- torchvision
- numpy
- tqdm
#### Dataset
Fashion-MNIST
- train: 60000张28*28图像
- test: 10000张28*28图像
#### Code
/src/MultilayerPerceptron.py
#### Result
| batch_size | lr | it_nums | layer_dims | train_loss | test_loss | test_acc |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| 64 | 0.01 | 20 | [28*28, 256, 128, 10] | 0.349 | 0.403 | 0.855 |
| 64 | 0.01 | 20 | [28*28, 256, 10] | 0.318 | 0.371 | 0.869 |

可以看出，相同超参数设置，mlp比单线性层效果要好，但是继续增加隐含层数量效果不一定正比增加

# Task02 02.13
## 1.文本预处理
文本是一类序列数据，一篇文章可以看作是字符或单词的序列，本节将介绍文本数据的常见预处理步骤，预处理通常包括四个步骤：

- 读入文本
- 分词，用现有工具进行分词(spaCy和NLTK)
- 建立字典，将每个词映射到一个唯一的索引（index）
- 将文本从词的序列转换为索引的序列，方便输入模型
## 2.语言模型
一段自然语言文本可以看作是一个离散时间序列，给定一个长度为 T 的词的序列 w1,w2,…,wT ，语言模型的目标就是评估该序列是否合理，即计算该序列的概率：
<center>P(w1,w2,…,wT)</center>    
词的概率可以通过该词在训练数据集中的相对词频来计算。序列长度增加，计算和存储多个词共同出现的概率的复杂度会呈指数级增加。 n 元语法通过马尔可夫假设简化模型，马尔科夫假设是指一个词的出现只与前面 n 个词相关，即 n 阶马尔可夫链（Markov Chain of Order  n ）

## 3.循环神经网络基础(RNN)
RNN目的是基于当前的输入与过去的输入序列，预测序列的下一个字符。循环神经网络引入一个隐藏变量 H ，用 Ht 表示 H 在时间步 t 的值。 Ht 的计算基于 Xt 和 Ht−1 ，可以认为 Ht 记录了到当前字符为止的序列信息，利用 Ht 对序列的下一个字符进行预测。

RNN:Ht=ϕ(XtWxh+Ht−1Whh+bh);Ot=HtWhq+bq.  

梯度裁剪：循环神经网络中较容易出现梯度衰减或梯度爆炸，这会导致网络几乎无法训练。裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。假设我们把所有模型参数的梯度拼接成一个向量  g ，并设裁剪的阈值是 θ 。裁剪后的梯度的 L2 范数不超过 θ 。  

one-hot编码：
- original: [1,2,3]
- one-hot: [[0,1,0,0],[0,0,1,0],[0,0,0,1]]   

### 实验
#### Dependencies
- torch
- torchvision
- numpy
- tqdm
#### Dataset
jaychou_lyrics.txt   
歌词数据集：周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词
#### Code
/src/nlp_tools.py
/src/rnn.py
#### Result
