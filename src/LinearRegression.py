import torch
import numpy as np 
import random
from tqdm import tqdm

def generate_dataset(dataset_size, inputs_size, true_weights, true_bais):
    assert inputs_size == len(true_weights)
    features = torch.randn(dataset_size, inputs_size, dtype=torch.float32)
    #print(features.size())
    true_weights = torch.FloatTensor(true_weights)
    true_weights = true_weights.view(true_weights.shape[0], 1)
    #print (true_weights.size())
    labels = torch.mm(features,  true_weights) + true_bais
    #print (labels.size())
    labels = labels + torch.FloatTensor(np.random.normal(0, 0.01, size=labels.size()))
    return [features, labels]

def data_iter(dataset, batch_size, shuffle=True):
    features, labels = dataset
    data_size = features.shape[0]
    data_indexs = list(range(data_size))
    if shuffle:
        np.random.shuffle(data_indexs)
    features, labels = features[data_indexs], labels[data_indexs]
    for i in range(0, data_size, batch_size):
        yield features[i:i+batch_size], labels[i:i+batch_size]

def init_weights_and_bais(weights_shape, bais_len=1):
    weights = torch.FloatTensor(np.random.uniform(0, 0.01, size=weights_shape))
    bais = torch.FloatTensor(np.ones([bais_len]))

    weights.requires_grad_(requires_grad=True)
    bais.requires_grad_(requires_grad=True)

    return weights, bais

def Linear(inputs, weights, bais):
    # print(inputs.shape, weights.shape, bais.shape)
    return torch.mm(inputs, weights) + bais

def squared_loss(y_pre, y_ture):
    return torch.sum(0.5*(y_pre-y_ture)**2)

def sgd(params, lr, batch_size):
    for param in params:
        param.data = param.data - lr*param.grad / batch_size

def train(dataset, lr, weights_shape, batch_size, it_nums):
    weights, bais = init_weights_and_bais(weights_shape)
    for i in range(it_nums):
        data_loader = data_iter(dataset, batch_size)
        train_loss = 0.
        for features, labels in tqdm(data_loader,  leave=False):
            outputs = Linear(features, weights, bais)
            loss = squared_loss(outputs, labels)
            train_loss = train_loss + loss.data.item()
            loss.backward()
            sgd([weights, bais], lr, batch_size)
            weights.grad.data.zero_()
            bais.grad.data.zero_()
        train_loss = train_loss / len(dataset[0])
        print(weights, bais)
        print('[iter{0}] train loss:{1}'.format(i+1, train_loss))
    

if __name__ == "__main__":
    data_size = 1000
    input_size = 3
    true_weights = [10.2, -33.4, 0.84]
    true_bais = 0.4
    lr = 0.03
    weights_shape = [3, 1]
    batch_size = 64
    it_nums = 100
    dataset = generate_dataset(data_size, input_size, true_weights, true_bais)
    train(dataset, lr, weights_shape, batch_size, it_nums)