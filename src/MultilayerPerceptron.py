import torch
import torchvision
import numpy as np 
from tqdm import tqdm

from SoftmaxClassification import generate_dataset, sgd, cross_entropy_softmax, Linear, init_weights_and_bais, data_iter, accuracy



def mlp(x, weights_list, bais_list, activation='relu'):
    assert  len(weights_list) == len(bais_list)

    if activation == 'relu':
        activation_f = torch.nn.ReLU()
    elif activation == 'tanh':
        activation_f = torch.nn.Tanh()
    elif activation == 'sigmoid':
        activation_f = torch.nn.Sigmoid()
    else:
        activation_f = torch.nn.ReLU()
    for weights, bais in zip(weights_list[:-1], bais_list[:-1]):

        x = torch.mm(x, weights) + bais
        x = activation_f(x)
    x = torch.mm(x, weights_list[-1]) + bais_list[-1]
    return x
def relu(x):
    return torch.max(x, torch.tensor(0.))

# def mlp(x, weights_list, bais_list):
#     h = relu(torch.mm(x, weights_list[0]) + bais_list[0])
#     return torch.mm(h, weights_list[1]) + bais_list[1]






def train(train_dataset, test_dataset, lr, in_dim, hidden_dims, out_dim, batch_size, it_nums):
    weights_list = list()
    bais_list = list()
    last_dim = in_dim
    for dim in hidden_dims:
        weights, bais = init_weights_and_bais((last_dim, dim), dim)
        weights_list.append(weights)
        bais_list.append(bais)
        last_dim = dim
    weights,bais = init_weights_and_bais((last_dim, out_dim), out_dim)
    weights_list.append(weights)
    bais_list.append(bais)
    
    
    for i in range(it_nums):
        train_loss = 0.
        train_dataloder = data_iter(train_dataset, batch_size, True)
        test_dataloder = data_iter(test_dataset, batch_size, False)
        for train_features, train_labels in tqdm(train_dataloder, leave=False):
            train_features = train_features.view(train_features.shape[0], -1)
            train_outputs = mlp(train_features, weights_list, bais_list)
            loss = cross_entropy_softmax(train_outputs, train_labels)
            train_loss = train_loss + loss.data.item()
            loss.backward()
            params = list()
            for weights_param, bais_param in zip(weights_list, bais_list):
                params.append(weights_param)
                params.append(bais_param)
            sgd(params, lr, batch_size)
            
            for param in params:
                param.grad.data.zero_()
            
        train_loss = train_loss / len(train_dataset[0])

        print('[iter{0}] train loss:{1}'.format(i+1, train_loss))
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for test_features, test_labels in tqdm(test_dataloder, leave=False):
                test_features = test_features.view(test_features.shape[0], -1)
                
                test_outputs = mlp(test_features, weights_list, bais_list)
                loss = cross_entropy_softmax(test_outputs, test_labels)
                acc = accuracy(test_outputs, test_labels)
                test_loss = test_loss + loss.data.item()
                test_acc = test_acc + acc
                 
        test_loss = test_loss / len(test_dataset[0])
        test_acc = test_acc / len(test_dataset[0])

        print('[iter{0}] test loss:{1} test acc:{2}'.format(i+1, test_loss, test_acc))

    
if __name__ == "__main__":
    lr = 0.03
    batch_size = 64
    in_dim = 28*28
    hidden_dims = [256]
    out_dim = 10
    it_nums = 20
    dataset_save_path = r'G:\xin.src\dive_into_deep_learning\dataset'
    download_dataset = True
    train_dataset, test_dataset = generate_dataset(dataset_save_path, download_dataset)
    train(train_dataset, test_dataset, lr, in_dim, hidden_dims, out_dim, batch_size, it_nums)

    

    
