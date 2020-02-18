import torch
import numpy as np
from tqdm import tqdm

from nlp_tools import generate_dataset, data_iter_random, data_iter_consecutive
from LinearRegression import init_weights_and_bais, sgd
from SoftmaxClassification import cross_entropy_softmax, to_onehot

# def get_params(input_dim, hidden_dim, output_dim):
#     input_weights, _ = init_weights_and_bais([input_dim, hidden_dim], hidden_dim)
#     hidden_weights, hidden_bais = init_weights_and_bais([hidden_dim, hidden_dim], hidden_dim)
#     output_weights, output_bais = init_weights_and_bais([hidden_dim, output_dim], output_dim)
#     return input_weights, hidden_weights, hidden_bais, output_weights, output_bais

def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        param = torch.zeros(shape, dtype=torch.float32)
        torch.nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs))
    return W_xh, W_hh, b_h, W_hq, b_q

def rnn(h, inputs, params, activation='tanh'):
    if activation == 'relu':
        activation_f = torch.nn.ReLU()
    elif activation == 'tanh':
        activation_f = torch.nn.Tanh()
    elif activation == 'sigmoid':
        activation_f = torch.nn.Sigmoid()
    else:
        activation_f = torch.nn.ReLU()
    outputs = list()
    for t in range(inputs.shape[-1]):
        x = inputs[:, :, t]
       
        h = activation_f(torch.mm(x, params[0])+torch.mm(h, params[1])+params[2])
        y = torch.mm(h, params[3])+params[4]
        outputs.append(y)
    return outputs

def init_hdden(batch_size, hidden_dim):
    return torch.Tensor(np.zeros((batch_size, hidden_dim)))


def predict_rnn(prefix, char_nums, params, hidden_dim, char_to_idx, idx_to_char, vocab_size):
    h = init_hdden(1, hidden_dim)
    outputs = [to_onehot(torch.Tensor([char_to_idx[prefix[0]]]), vocab_size)]

    char_nums = 50
    for t in range(len(prefix)+char_nums-1):
        x = outputs[-1]

        x = x.unsqueeze(2)
        y = rnn(h, x, params)
        if t < len(prefix)-1: 
            outputs.append(to_onehot(torch.Tensor([char_to_idx[prefix[t+1]]]), vocab_size))
        else:
            outputs.append(y[0])

    outputs = ''.join([idx_to_char[torch.argmax(i)] for i in outputs])
    return outputs

def clip_grad(params,theta):
    normal = torch.Tensor([0.])
    for param in params:
        normal = normal+torch.sum(param.grad.data**2)
    normal = torch.sqrt(normal).item()
    if normal > theta:
        for param in params:
            param.grad.data *= (theta / normal)


def train(corpus_words, char_to_idx, idx_to_char,  batch_size, lr, it_nums, vocab_size, hidden_dim, char_nums, prefixs, sample_batch='random'):
    params = get_params(vocab_size, hidden_dim, vocab_size)
    cross_entropy =  torch.nn.CrossEntropyLoss()
    for i in range(it_nums):
        # print(params)
        if sample_batch == 'random':
            data_loader = data_iter_random(corpus_words, batch_size, char_nums)
        elif sample_batch == 'consecutive':
            data_loader = data_iter_consecutive(corpus_words, batch_size, char_nums)
        else:
            data_loader = data_iter_random(corpus_words, batch_size, char_nums)
        n = 0
        train_loss = 0.
        for x, y in tqdm(data_loader, leave=False):
            h = init_hdden(batch_size, hidden_dim)
            x = torch.stack([to_onehot(x[:, i], vocab_size) for i in range(x.shape[1])], dim=2)
        
            # y = torch.stack([to_onehot(y[:, i], vocab_size) for i in range(y.shape[1])], dim=2)
            outputs = rnn(h, x, params)
            outputs = torch.stack(outputs, dim=2)
            outputs = outputs.permute(0, 2, 1)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            # print(outputs.shape)
            # y = y.permute(0, 2, 1)
            y = y.contiguous().view(-1)
            # print(outputs, y)
            loss = cross_entropy(outputs, y.long())
            train_loss = train_loss + loss.data.item()*y.shape[0]
            loss.backward()
            clip_grad(params, 0.01)
            sgd(params, lr, 1)
            for param in params:
                param.grad.data.zero_()
            n = n+ y.shape[0]

        perplexity = np.exp(train_loss / n)
        print('[iter{0}] perplexity:{1}'.format(i+1, perplexity))
        for prefix in prefixs:
            sentence = predict_rnn(prefix, char_nums, params, hidden_dim, char_to_idx, idx_to_char, vocab_size)
            print('[iter{0}] sentence:{1}'.format(i+1, sentence))



if __name__ == "__main__":
    batch_size = 32
    lr = 100
    it_nums = 300
    hidden_dim = 256
    char_nums = 35
    prefixs = ['分开', '什么']
    corpus_chars, corpus_words, idx_to_char, char_to_idx, vocab_size = generate_dataset(r'G:\xin.src\dataset\dataset\jaychou_lyrics/jaychou_lyrics.txt')
    train(corpus_words, char_to_idx, idx_to_char, batch_size, lr, it_nums, vocab_size, hidden_dim, char_nums, prefixs)
            
            


    

        
    
